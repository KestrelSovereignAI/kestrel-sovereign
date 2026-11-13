"""Contract tests for cancellation-safe ownership and borrowed observation."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import kestrel_sovereign
from kestrel_sovereign._async_ownership import (
    OwnedAsyncIterator,
    OwnedTaskAdmission,
    acquire_lock_keeping_cancellation,
    await_owned_task,
    create_owned_task,
    owned_task_error,
    owned_iterator_parent_tasks,
    raise_owned_outcome,
    run_blocking_operation,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
@pytest.mark.parametrize("mode", ["success", "allocation", "callback-before", "callback-after"])
async def test_original_admission_cell_precedes_all_callback_publication(
    tmp_path, mode, eager,
):
    import inspect
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from tests.unit.test_host_context_native_cleanup import _actual_operation
    path = tmp_path / "original-task-admission.db"
    entered, release = asyncio.Event(), asyncio.Event()
    body_started = []
    async def original():
        body_started.append(asyncio.current_task())
        db = await AsyncDatabase.sqlite(str(path))
        try:
            await db.execute("CREATE TABLE admission_effects(id INTEGER)")
            entered.set()
            await release.wait()
            return "original native result"
        finally:
            await db.close()
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    admission, accepted, refused = OwnedTaskAdmission(), [], []
    refusal = RuntimeError("actual canonical admission " + mode)
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if mode.startswith("callback") and getattr(callback, "__name__", "") == "close_unstarted":
                assert admission.task is self, "accepted Task was invisible during callback publication"
                assert not body_started, "eager body ran before canonical publication"
                if mode.endswith("-after"):
                    super().add_done_callback(callback, context=context)
                raise refusal
            return super().add_done_callback(callback, context=context)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is original.__code__:
            if mode == "allocation":
                refused.append((coroutine, operation))
                raise refusal
            options = {"eager_start": True} if eager else {}
            task = PublicationTask(coroutine, loop=loop, **kwargs, **options)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    try:
        loop.set_task_factory(factory)
        if mode == "success":
            task = create_owned_task(original(), admission=admission, retain_originals=True)
            assert admission.task is task
            assert not body_started
        else:
            with pytest.raises(RuntimeError) as caught:
                create_owned_task(original(), admission=admission, retain_originals=True)
            if mode == "allocation":
                from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
                # An opaque factory may have accepted an outer-await wrapper.
                # Keep its source error and explicit debt, never infer absence.
                assert refusal in _explicit_control_chain(caught.value)
                assert admission._acceptance_unknown
                assert execution_cleanup_debts(caught.value)
            else:
                assert caught.value is refusal
        loop.set_task_factory(previous)
        if mode == "success":
            await entered.wait()
            release.set()
            assert (await await_owned_task(task)).result == "original native result"
        elif admission.task is not None:
            assert admission.task is accepted[0]
            assert (await await_owned_task(admission.task)).error is refusal
            assert not path.exists()
        else:
            assert not accepted and not path.exists()
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
    finally:
        loop.set_task_factory(previous)
        release.set()
        for task in accepted:
            await await_owned_task(task)


@pytest.mark.asyncio
async def test_borrowed_observation_withdraws_without_cancelling_or_joining_original():
    entered, release = asyncio.Event(), asyncio.Event()

    async def original():
        await release.wait()
        return "original result"

    source = create_owned_task(original(), retain_originals=True)

    async def observe():
        entered.set()
        return await await_owned_task(source, defer_cancellation=False)

    observer = create_owned_task(observe(), retain_originals=True)
    try:
        await entered.wait()
        observer.cancel("actual borrowed observer withdrew")
        observed = await await_owned_task(observer)
        assert isinstance(observed.error, asyncio.CancelledError)
        assert observed.error.args == ("actual borrowed observer withdrew",)
        assert not source.done() and not source.cancelling()
        release.set()
        assert (await await_owned_task(source)).result == "original result"
    finally:
        release.set()
        await await_owned_task(observer)
        await await_owned_task(source)


@pytest.mark.asyncio
async def test_borrowed_callback_removal_failure_cannot_demote_caller_cancellation():
    from kestrel_sovereign.execution_custody import _explicit_control_chain
    entered, release = asyncio.Event(), asyncio.Event()
    refusal = RuntimeError('actual borrowed completion callback removal refused')
    loop = asyncio.get_running_loop()
    factory = loop.get_task_factory()
    observer = None
    class RejectRemoval(asyncio.Task):
        def remove_done_callback(self, callback):
            if getattr(callback, '__name__', '') == 'mark_complete' and asyncio.current_task() is observer:
                raise refusal
            return super().remove_done_callback(callback)
    async def original():
        await release.wait()
        return 'genuine original result'
    loop.set_task_factory(lambda loop, coro, **kwargs: RejectRemoval(coro, loop=loop, **kwargs))
    try:
        source = create_owned_task(original())
    finally:
        loop.set_task_factory(factory)
    async def borrow():
        entered.set()
        return await await_owned_task(source, defer_cancellation=False)
    observer = create_owned_task(borrow())
    try:
        await entered.wait()
        observer.cancel('actual borrowed caller withdrawal with removal failure')
        observed = await await_owned_task(observer)
        assert observed.error is not None and not isinstance(observed.error, Exception)
        chain = tuple(_explicit_control_chain(observed.error))
        assert any(error is refusal for error in chain)
        assert any(isinstance(error, asyncio.CancelledError) and error.args == ('actual borrowed caller withdrawal with removal failure',) for error in chain)
        assert not source.done() and not source.cancelling()
    finally:
        release.set()
        await await_owned_task(observer)
        assert (await await_owned_task(source)).result == 'genuine original result'


@pytest.mark.asyncio
@pytest.mark.parametrize("allocation", ["completion", "observation"])
async def test_completion_allocation_refusal_cannot_abandon_original(monkeypatch, allocation):
    from tests.shared.owned_task_observation import refuse_owned_observation_allocation

    release = asyncio.Event()
    failure = RuntimeError("controlled original completion allocation refusal")

    async def original():
        await release.wait()
        return "original result"

    source = create_owned_task(original(), retain_originals=True)
    refused = refuse_owned_observation_allocation(monkeypatch, lambda: source,
        allocation=allocation, error=failure)
    observer = asyncio.create_task(await_owned_task(source))
    try:
        await asyncio.wait_for(refused.wait(), 2)
        await asyncio.sleep(0.02)
        assert not observer.done(), "allocation refusal abandoned the original producer"
        assert not source.done() and not source.cancelling()
        release.set()
        outcome = await observer
        assert source.result() == "original result"
        assert outcome.error is not None
        assert "completion allocation refusal" in str(outcome.error)
    finally:
        release.set()
        await asyncio.gather(observer, return_exceptions=True)
        await await_owned_task(source)


@pytest.mark.asyncio
async def test_bounded_wait_observer_preserves_unbounded_and_first_completed(monkeypatch):
    from tests.shared.owned_task_observation import observe_bounded_asyncio_wait

    loop = asyncio.get_running_loop()
    ready, pending = loop.create_future(), loop.create_future()
    ready.set_result('original result')
    observations = []
    entries = []

    async def before_wait(owners, timeout):
        observations.append((set(owners), timeout))

    observe_bounded_asyncio_wait(monkeypatch, before_wait,
        on_enter=lambda owners: entries.append(tuple(owners)))
    try:
        async with asyncio.timeout(1):
            done, waiting = await asyncio.wait(iter((ready, pending)),
                return_when=asyncio.FIRST_COMPLETED)
        assert done == {ready} and waiting == {pending}
        assert entries == [(ready, pending)]
        assert ready.result() == 'original result' and observations == []
        done, waiting = await asyncio.wait({pending}, timeout=0.01)
        assert done == set() and waiting == {pending} and not pending.cancelled()
        assert observations == [({pending}, 0.01)]
    finally:
        pending.set_result('original pending result')


@pytest.mark.asyncio
async def test_borrowed_observation_returns_original_cancelled_source_not_shield_surrogate():
    original = asyncio.CancelledError("original source cancellation")
    cause = OSError("original source cause")

    async def exit():
        raise original from cause

    source = create_owned_task(exit(), retain_originals=True)
    await await_owned_task(source)
    observed = await await_owned_task(source, defer_cancellation=False)
    assert observed.error is original and observed.error.__cause__ is cause
    assert observed.cancellation is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pending", "forwarding"])
async def test_borrowed_observation_refuses_owning_or_forwarding_cancellation(mode):
    source = asyncio.get_running_loop().create_future()
    kwargs = ({"pending_cancellation": asyncio.CancelledError("already owned")}
              if mode == "pending" else {"cancel_on_caller_cancellation": True})
    with pytest.raises(ValueError, match="borrowed observation"):
        await await_owned_task(source, defer_cancellation=False, **kwargs)
    assert not source.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("retain_originals", [False, True])
async def test_native_original_policy_keeps_caused_cancellation_without_changing_default(retain_originals):
    cause = ValueError("original native cause")
    cancellation = asyncio.CancelledError("original native cancellation")

    async def exit():
        raise cancellation from cause

    task = create_owned_task(exit(), retain_originals=retain_originals)
    outcome = await await_owned_task(task)
    expected = cancellation if retain_originals else cause
    assert outcome.error is expected
    assert owned_task_error(task) is expected
    assert (await await_owned_task(task)).error is expected
    if retain_originals:
        assert cancellation.__cause__ is cause


@pytest.mark.asyncio
@pytest.mark.parametrize("retain_originals", [False, True])
async def test_original_policy_applies_to_native_outer_completion(tmp_path, retain_originals):
    cause = ValueError("original physical epilogue failure")
    cancellation = asyncio.CancelledError("original physical outer completion")
    effect = tmp_path / "original-native-outer-effect"
    owners = []
    async def operation():
        effect.write_text("original effect completed once")
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **options):
        async def physical():
            await coroutine
            raise cancellation from cause
        task = asyncio.Task(physical(), loop=loop, **options)
        owners.append(task)
        return task
    try:
        loop.set_task_factory(factory)
        owner = create_owned_task(operation(), retain_originals=retain_originals)
        loop.set_task_factory(previous)
        outcome = await await_owned_task(owner)
        assert outcome.error is (cancellation if retain_originals else cause), (
            "Original admission policy was lost at the native outer completion")
        assert (await await_owned_task(owner)).error is outcome.error
        assert effect.read_text() == "original effect completed once" and len(owners) == 1 and owner.done()
    finally:
        loop.set_task_factory(previous)
        for owner in owners:
            await await_owned_task(owner)


@pytest.mark.asyncio
async def test_only_explicit_owned_iterator_sources_inherit_observation_lineage():
    consumer = asyncio.current_task()
    sources = []

    async def unrelated():
        return owned_iterator_parent_tasks()

    async def inner():
        parents = owned_iterator_parent_tasks()
        assert parents == (sources[0], consumer)
        assert await asyncio.create_task(unrelated()) == ()
        assert await asyncio.to_thread(owned_iterator_parent_tasks) == ()
        yield "joined"

    async def outer():
        sources.append(asyncio.current_task())
        assert owned_iterator_parent_tasks() == (consumer,)
        iterator = OwnedAsyncIterator(inner, operation="nested observation")
        try:
            yield await anext(iterator)
        finally:
            await iterator.aclose()

    iterator = OwnedAsyncIterator(outer, operation="root observation")
    try:
        assert await anext(iterator) == "joined"
        assert owned_iterator_parent_tasks() == ()
    finally:
        await iterator.aclose()


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
@pytest.mark.parametrize('bounded', [False, True])
async def test_repeated_withdrawal_does_not_accumulate_completion_callbacks(tmp_path, bounded):
    """Cancellation must not grow the original owner's observation inventory."""
    from kestrel_sovereign._async_ownership import wait_owned_or_pending
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    release = asyncio.Event()
    database = await AsyncDatabase.sqlite(str(tmp_path / 'original-owner.db'))
    await database.execute('CREATE TABLE original_receipt(value TEXT)')

    async def original():
        await release.wait()
        await database.execute("INSERT INTO original_receipt VALUES ('settled')")
        return 'settled'

    source = create_owned_task(original(), retain_originals=True)
    operation = (wait_owned_or_pending(source, cancellation_grace=10,
        cancel_on_caller_cancellation=False) if bounded else await_owned_task(source))
    observer = asyncio.create_task(operation)
    try:
        await asyncio.sleep(0)
        # Inspect the real completion Future, without replacing allocation,
        # cancellation delivery, or the native database operation.
        awaited = observer.get_coro()
        while awaited.cr_code.co_name not in {'await_owned_task', 'wait_owned_or_pending'}:
            awaited = awaited.cr_await
        completion = awaited.cr_frame.f_locals['completion']
        for number in range(128):
            observer.cancel(f'withdrawal-{number}')
            await asyncio.sleep(0)
        assert not source.done() and not source.cancelling()
        assert len(completion._callbacks or ()) <= 2, (
            'each withdrawal retained another callback on the original completion')
    finally:
        release.set()
        outcome = await observer
        try:
            assert outcome.error is None and outcome.result == 'settled'
            assert outcome.cancellation.args == ('withdrawal-0',)
            row = await database.fetchone('SELECT COUNT(*) FROM original_receipt')
            assert row[0] == 1
        finally:
            await database.close()


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
@pytest.mark.parametrize("already_stopping", [False, True])
@pytest.mark.parametrize("falsey", [False, True])
async def test_owned_cancellation_forwarder_uses_original_owner_once(already_stopping, falsey):
    entered, unwinding, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    forwards = []

    async def operation():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            unwinding.set()
            await release.wait()

    task = create_owned_task(operation())

    def forward(original):
        assert original is task
        forwards.append(original)
        if not already_stopping:
            original.cancel()

    class FalseyForwarder:
        def __bool__(self):
            return False

        def __call__(self, original):
            forward(original)

    policy = FalseyForwarder() if falsey else forward
    owner = create_owned_task(await_owned_task(task, cancel_on_caller_cancellation=policy))
    try:
        await entered.wait()
        if already_stopping:
            task.cancel()
            await asyncio.wait_for(unwinding.wait(), 1)
        owner.cancel("original caller stop")
        await asyncio.wait_for(unwinding.wait(), 1)
        await asyncio.sleep(0)
        owner.cancel("repeated caller stop")
        await asyncio.sleep(0)
        assert not owner.done(), "forwarding interrupted the original unwind"
        release.set()
        result = (await await_owned_task(owner)).result
        assert forwards == [task]
        assert isinstance(result.error, asyncio.CancelledError)
        assert result.cancellation.args == ("original caller stop",)
        assert task.cancelling() == 1
    finally:
        release.set()
        if not task.done() and not task.cancelling():
            task.cancel()
        await await_owned_task(task)
        await await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("control", [False, True])
async def test_forwarder_failure_preserves_original_owned_error(control):
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from tests.utils.original_failure import OriginalFailure

    entered, release, forwarded = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = ExecutionAuthorityError("original work denied") if control else LookupError("original work failed")
    source = OriginalFailure.capture(original)
    forwarding_failure = ValueError("original forwarding failure")

    async def operation():
        entered.set()
        await release.wait()
        raise original

    def forward(_task):
        forwarded.set()
        raise forwarding_failure

    task = create_owned_task(operation())
    owner = create_owned_task(await_owned_task(task, cancel_on_caller_cancellation=forward))
    try:
        await entered.wait()
        owner.cancel("original caller stop")
        await asyncio.wait_for(forwarded.wait(), 1)
        assert not owner.done()
        release.set()
        result = (await await_owned_task(owner)).result
        source.assert_retained(result.error)
        if not control:
            assert result.error is original
        assert any(str(forwarding_failure) in note for note in result.error.__notes__)
        assert result.cancellation.args == ("original caller stop",)
    finally:
        release.set()
        await await_owned_task(task)
        await await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("control", [False, True])
async def test_failed_cancellation_forwarder_still_physically_joins(control):
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError

    entered, release, forwarded = asyncio.Event(), asyncio.Event(), asyncio.Event()
    failure = ExecutionAuthorityError("forwarder denied") if control else ValueError("forwarder failed")

    async def operation():
        entered.set()
        await release.wait()
        return "physically settled"

    def forward(_task):
        forwarded.set()
        raise failure

    task = create_owned_task(operation())
    owner = create_owned_task(await_owned_task(task, cancel_on_caller_cancellation=forward))
    try:
        await entered.wait()
        owner.cancel("original caller stop")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not owner.done(), "a failed forwarder abandoned the owned operation"
        assert forwarded.is_set()
        release.set()
        result = (await await_owned_task(owner)).result
        assert result.error is failure and task.result() == "physically settled"
        assert result.cancellation.args == ("original caller stop",)
        if control:
            with pytest.raises(ExecutionAuthorityError) as caught:
                raise_owned_outcome(result, operation="forwarding test")
            assert caught.value is failure
        else:
            with pytest.raises(asyncio.CancelledError) as caught:
                raise_owned_outcome(result, operation="forwarding test")
            assert caught.value.__cause__ is failure
    finally:
        release.set()
        await await_owned_task(task)
        await await_owned_task(owner)


_TEARDOWN_SCENARIO = """
import asyncio

from kestrel_sovereign._async_ownership import await_owned_task


async def owned():
    try:
        await asyncio.sleep(30)
    except asyncio.CancelledError:
        for _ in range(3):  # unwinds over several loop iterations
            await asyncio.sleep(0)
        raise


async def owner():
    return await await_owned_task(asyncio.create_task(owned()))


async def main():
    owner_task = asyncio.create_task(owner())
    await asyncio.sleep(0.01)
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()
    await asyncio.wait({owner_task}, timeout=5)
    print("owner done:", owner_task.done())


asyncio.run(main())
"""


def test_teardown_cancelling_every_task_does_not_spin_the_owner():
    """Regression: event-loop teardown cancels every task, including the waiter
    task this helper used to create. A cancelled waiter made each ``shield``
    raise without yielding, so the owner spun and the owned task never ran
    again. A spinning loop cannot be interrupted from inside, so the scenario
    runs in a subprocess that is killed if it does not finish."""
    # Only the directory holding this package: other tests put directories on
    # sys.path that shadow the standard library in a fresh interpreter.
    package_root = Path(kestrel_sovereign.__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(package_root)}
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _TEARDOWN_SCENARIO],
            capture_output=True, text=True, timeout=30, env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the owner spun instead of yielding to the event loop")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "owner done: True"


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
@pytest.mark.parametrize('watch_started', [False, True])
async def test_cancelling_unstarted_iterator_owner_interrupts_source(watch_started):
    """The same cancellation owner retires its source before or after startup."""

    entered = asyncio.Event()
    finalized = asyncio.Event()

    async def source():
        try:
            entered.set()
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            finalized.set()

    owned = OwnedAsyncIterator(
        source,
        operation="never-started watcher",
        cleanup_requested=lambda: True,
    )
    if watch_started:
        await entered.wait()
        await asyncio.sleep(0)
    # Registration can expose the watcher before its first step, or Stop can
    # arrive while its original borrowed observation is already suspended.
    from kestrel_sovereign._async_ownership import cancel_owned_task
    assert cancel_owned_task(owned.owner_task) is True

    await asyncio.wait_for(entered.wait(), timeout=0.2)
    for _attempt in range(20):
        if finalized.is_set():
            break
        await asyncio.sleep(0)

    assert finalized.is_set()
    assert owned.interrupted_by_cleanup is True
    await owned.aclose()


@pytest.mark.asyncio
async def test_stop_marker_does_not_relabel_natural_unwind_as_cleanup():
    """A Stop marker cannot claim an independent terminal source failure."""

    stop_requested = False

    async def source():
        try:
            yield "last item"
        finally:
            raise RuntimeError("natural unwind cleanup failed")

    owned = OwnedAsyncIterator(
        source,
        operation="naturally unwound source",
        cleanup_requested=lambda: stop_requested,
    )
    assert await anext(owned) == "last item"
    stop_requested = True

    with pytest.raises(RuntimeError, match="natural unwind cleanup failed"):
        await anext(owned)

    assert owned.cleanup_error is None


@pytest.mark.asyncio
async def test_natural_source_failure_is_not_cleanup_debt():
    """Only an initiated close, not ordinary iteration, owns cleanup debt."""

    async def source():
        try:
            yield "last item"
        finally:
            raise RuntimeError("natural unwind cleanup failed")

    owned = OwnedAsyncIterator(
        source,
        operation="naturally unwound source",
    )
    assert await anext(owned) == "last item"

    with pytest.raises(RuntimeError, match="natural unwind cleanup failed"):
        await anext(owned)

    assert owned.cleanup_error is None


@pytest.mark.asyncio
async def test_initiated_close_still_records_source_unwind_failure_as_cleanup():
    """The real close boundary retains failures from cancellation-driven unwind."""

    entered = asyncio.Event()

    async def source():
        try:
            entered.set()
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            raise RuntimeError("close unwind cleanup failed")

    owned = OwnedAsyncIterator(source, operation="failed close")
    consumer = asyncio.create_task(anext(owned))
    await entered.wait()

    with pytest.raises(RuntimeError, match="close unwind cleanup failed"):
        await owned.aclose()
    # Physical close also joins the original cancellation observer; delivery
    # of its terminal notification to this reader is a separate scheduling
    # step. Prove the reader is terminal before testing post-terminal cancel,
    # rather than racing a genuine caller withdrawal into that native join.
    done, _ = await asyncio.wait({consumer}, timeout=1)
    assert done, "original cleanup failure did not reach its terminal reader"
    assert consumer.cancel() is False
    with pytest.raises(RuntimeError, match="close unwind cleanup failed"):
        await consumer

    assert isinstance(owned.cleanup_error, RuntimeError)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["plain", "sdk", "audit", "cancelled-future"])
async def test_unadmitted_native_cancellation_has_one_stable_terminal_observation(kind):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, _explicit_control_chain
    original = asyncio.CancelledError("unadmitted native original cancellation")
    if kind in {"sdk", "audit"}:
        original.__cause__ = (ResourceOutcomeUnknown("unadmitted original effect acknowledgement")
            if kind == "sdk" else ToolAuditUnconfirmedError("unadmitted original audit acknowledgement"))
    async def source():
        raise original
    if kind == "cancelled-future":
        task = asyncio.get_running_loop().create_future()
        task.cancel("unadmitted native Future cancellation")
    else:
        task = asyncio.create_task(source())
    first = await await_owned_task(task)
    if kind != "cancelled-future":
        assert first.error is original
    original_graph = tuple(_explicit_control_chain(first.error))
    for _ in range(3):
        again = await await_owned_task(task)
        assert again.error is first.error, "Repeated native observation manufactured another cancellation"
        assert owned_task_error(task) is first.error
        assert tuple(_explicit_control_chain(first.error)) == original_graph


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "sdk", "audit", "plain"])
async def test_private_cancellation_is_bound_to_its_original_admission_and_carrier(kind):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign._async_ownership import cancel_owned_task, is_owned_task_cancellation
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, execution_observation_error
    entered = asyncio.Event()
    async def source():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            if kind is not None:
                error.__cause__ = (ResourceOutcomeUnknown("original requested cancellation retains effect uncertainty")
                    if kind == "sdk" else ToolAuditUnconfirmedError("original requested cancellation retains audit uncertainty")
                    if kind == "audit" else RuntimeError("independent explicit source failure"))
            raise
    owner = create_owned_task(source(), retain_originals=True)
    await entered.wait()
    assert cancel_owned_task(owner)
    outcome = await await_owned_task(owner)
    assert is_owned_task_cancellation(owner, outcome.error) is (kind is None)
    assert not is_owned_task_cancellation(asyncio.current_task(), outcome.error)
    assert not is_owned_task_cancellation(owner, asyncio.CancelledError("independent cancellation")), (
        "Independent cancellation became an authenticated original private Stop")
    assert (await await_owned_task(owner)).error is outcome.error
    if kind is None:
        combined = execution_observation_error(outcome.error,
            asyncio.CancelledError("independent original cancellation"), prefer_cancellation=True)
        assert not is_owned_task_cancellation(owner, combined), (
            "Combined original evidence became an authenticated private Stop")


def test_native_cancellation_helpers_reject_invalid_input():
    from kestrel_sovereign._async_ownership import cancel_owned_task, owned_task_completion_error
    with pytest.raises(TypeError, match="original native Task"):
        cancel_owned_task(object())
    with pytest.raises(TypeError, match="requires BaseException or None"):
        owned_task_completion_error(object(), object())


@pytest.mark.asyncio
async def test_native_cancellation_recovery_never_uses_exception_truthiness():
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign._async_ownership import owned_task_completion_error
    class FalsyCancellation(asyncio.CancelledError):
        def __bool__(self):
            return False
    original = FalsyCancellation("original native cancellation with false truth value")
    original.__cause__ = ResourceOutcomeUnknown("original acknowledgement remains uncertain")
    async def source():
        raise original
    owner = create_owned_task(source(), retain_originals=True)
    try:
        await owner
    except asyncio.CancelledError as received:
        assert owned_task_completion_error(owner, received) is original, (
            "Native cancellation recovery replaced an original because its truth value was false")
    else:
        pytest.fail("Original native cancellation disappeared")


@pytest.mark.asyncio
@pytest.mark.parametrize("withdraw_during_restoration", [False, True])
async def test_native_cancellation_recovery_preserves_actual_writer_restoration_withdrawal(
    tmp_path, withdraw_during_restoration,
):
    from kestrel_sovereign._async_ownership import owned_native_wait, owned_task_completion_error, owned_task_error
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError, _explicit_control_chain
    from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock
    lock = ReentrantTransitionLock()
    entered, release = asyncio.Event(), asyncio.Event()
    original = asyncio.CancelledError("original source failed while descendant owns writer")
    original.__cause__ = ExecutionCommitOutcomeError("unknown")
    original_graph = tuple(_explicit_control_chain(original))
    token = object()
    tasks, received_errors = [], []
    effect = tmp_path / "actual-descendant-writer-effect"
    async def writer():
        async with lock:
            entered.set()
            await release.wait()
            effect.write_text("original writer completed once")
    async def child():
        tasks.append(create_owned_task(writer(), retain_originals=True))
        await entered.wait()
        raise original
    async def parent():
        async with lock.policy_span():
            async with lock:
                source = create_owned_task(child(), retain_originals=True)
                tasks.append(source)
                try:
                    async with owned_native_wait((source,)):
                        await source
                except asyncio.CancelledError as received:
                    received_errors.append(received)
                    recovered = owned_task_completion_error(source, received)
                    assert lock._writer is asyncio.current_task(), "Cancellation escaped before original writer restoration"
                    return source, recovered
                pytest.fail("Original native cancellation disappeared")
    owner = asyncio.create_task(parent())
    try:
        async with asyncio.timeout(1):
            while not (lock._parked and lock._parked[-1].restoring):
                await asyncio.sleep(0)
        assert not effect.exists() and not owner.done()
        if withdraw_during_restoration:
            owner.cancel(token)
            await asyncio.sleep(0)
            assert not owner.done(), "Restoration cancellation detached the actual descendant writer"
        release.set()
        source, recovered = await owner
        assert owned_task_error(source) is original
        assert original in tuple(_explicit_control_chain(recovered))
        if withdraw_during_restoration:
            actual = [error for error in _explicit_control_chain(received_errors[0])
                if isinstance(error, asyncio.CancelledError) and len(error.args) == 1 and error.args[0] is token]
            assert len(actual) == 1
            assert actual[0] in tuple(_explicit_control_chain(recovered)), (
                "Native cancellation recovery discarded actual writer-restoration withdrawal")
        assert tuple(_explicit_control_chain(original)) == original_graph
        assert effect.read_text() == "original writer completed once" and lock._writer is None
        assert all(task.done() for task in tasks)
    finally:
        release.set()
        await asyncio.gather(owner, *tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("handling_error", [False, True])
@pytest.mark.parametrize("mode", [
    pytest.param("eager", marks=pytest.mark.skipif(sys.version_info < (3, 12), reason="native eager Task requires Python 3.12")),
    pytest.param("eager-preconsumed", marks=pytest.mark.skipif(sys.version_info < (3, 12), reason="native eager Task requires Python 3.12")),
    "prior-callback", "canonical-prior-callback",
])
@pytest.mark.parametrize("kind", [None, "plain", "sdk", "audit"])
async def test_original_completion_capture_cannot_rely_on_callback_registration(tmp_path, mode, kind, handling_error):
    import inspect
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign._async_ownership import OwnedTaskAdmission, owned_task_completion_error, owned_task_error
    from kestrel_sovereign.execution_custody import (
        ToolAuditUnconfirmedError, _explicit_control_chain, execution_cleanup_debts,
        is_tool_execution_terminal_error,
    )
    original = asyncio.CancelledError("original outer completion before capture")
    if kind in {"sdk", "audit"}:
        original.__cause__ = (ResourceOutcomeUnknown("original outer effect uncertain") if kind == "sdk"
            else ToolAuditUnconfirmedError("original outer audit uncertain"))
    effect = tmp_path / "original-outer-completion-effect"
    bodies, consumed, owners = [], [], []
    async def operation():
        bodies.append(asyncio.current_task())
        effect.write_text("original physical effect completed once")
        if kind is not None:
            raise original
        return "original outer success"
    payload = operation()
    admission = OwnedTaskAdmission()
    caller_error = RuntimeError("unrelated caller exception during original admission")
    def admit():
        return create_owned_task(payload, admission=admission, retain_originals=True)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def consume(task):
        if mode == "canonical-prior-callback":
            task._kestrel_owned_terminal_error = None
        try:
            task.exception()
        except asyncio.CancelledError as error:
            consumed.append(error)
    def factory(loop, coroutine, **options):
        if mode == "canonical-prior-callback":
            task = asyncio.Task(coroutine, loop=loop, **options)
            owners.append(task)
            task.add_done_callback(consume)
            return task
        async def outer():
            if mode == "prior-callback":
                await asyncio.sleep(0)
            coroutine.close()
            effect.write_text("original physical effect completed once")
            if kind is not None:
                raise original
            return "original outer success"
        if mode.startswith("eager"):
            options["eager_start"] = True
        task = asyncio.Task(outer(), loop=loop, **options)
        owners.append(task)
        if mode == "prior-callback":
            task.add_done_callback(consume)
        elif mode == "eager-preconsumed":
            consume(task)
        return task
    try:
        loop.set_task_factory(factory)
        if handling_error:
            try:
                raise caller_error
            except RuntimeError:
                owner = admit()
        else:
            owner = admit()
        loop.set_task_factory(previous)
        try:
            value = await owner
        except asyncio.CancelledError as received:
            recovered = owned_task_completion_error(owner, received)
        else:
            assert kind is None and value == "original outer success"
            recovered = owned_task_error(owner)
        if kind is None:
            assert recovered is None
        elif mode in {"eager", "canonical-prior-callback"}:
            assert recovered is original, (
                "Scheduled completion capture lost an already-terminal original before native await")
            assert is_tool_execution_terminal_error(recovered) is (kind in {"sdk", "audit"})
        else:
            assert consumed and consumed[0] is original
            assert is_tool_execution_terminal_error(recovered), (
                "Registered completion callback falsely certified a consumed original cancellation")
            assert any(task is owner for _, task in execution_cleanup_debts(recovered)), (
                "Unobservable original completion lost its exact native owner debt")
            assert original not in tuple(_explicit_control_chain(recovered)), "Lost original identity was fabricated"
        assert owned_task_error(owner) is recovered
        assert caller_error not in tuple(_explicit_control_chain(recovered))
        assert (await await_owned_task(owner)).error is recovered
        assert effect.read_text() == "original physical effect completed once"
        assert len(bodies) == int(mode == "canonical-prior-callback")
        assert len(owners) == 1 and owner.done()
        assert inspect.getcoroutinestate(payload) == inspect.CORO_CLOSED
    finally:
        loop.set_task_factory(previous)
        for owner in owners:
            await await_owned_task(owner)
        payload.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "ordinary", "sdk", "audit"])
@pytest.mark.parametrize("observer_fails", [False, True])
async def test_native_completion_retains_independent_observer_io_failure(tmp_path, kind, observer_fails):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign._async_ownership import owned_native_completion
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, _explicit_control_chain
    source = RuntimeError("original native physical failure") if kind == "ordinary" else None
    if kind in {"sdk", "audit"}:
        source = asyncio.CancelledError("original native physical cancellation")
        source.__cause__ = (ResourceOutcomeUnknown("original native effect uncertain") if kind == "sdk"
            else ToolAuditUnconfirmedError("original native audit uncertain"))
    effect = tmp_path / "original-native-source-effect"
    async def operation():
        effect.write_text("original native source physically completed")
        if source is not None:
            raise source
        return "original physical success"
    owner = create_owned_task(operation(), retain_originals=True)
    handle = (tmp_path / "actual-observer-receipt").open("w")
    observer_errors = []
    observed, value = None, None
    try:
        with owned_native_completion(owner):
            try:
                value = await owner
            finally:
                handle.close()
                if observer_fails:
                    try:
                        handle.write("receipt after actual close")
                    except ValueError as error:
                        observer_errors.append(error)
                        raise
    except BaseException as error:
        observed = error
    finally:
        handle.close()
        await await_owned_task(owner)
    chain = tuple(_explicit_control_chain(observed))
    if source is not None:
        assert source in chain
    if observer_fails:
        assert len(observer_errors) == 1 and observer_errors[0] in chain, (
            "Original native completion erased independent observer I/O failure")
    elif source is None:
        assert observed is None and value == "original physical success"
    assert owner.done() and handle.closed
    assert effect.read_text() == "original native source physically completed"


@pytest.mark.asyncio
async def test_acquire_lock_keeping_cancellation_returns_every_cancellation_once_held():
    """#4040: a closer cancelled while it waits for a lock still gets the
    lock, and the cancellation is handed back rather than raised."""
    lock = asyncio.Lock()
    await lock.acquire()
    acquiring = asyncio.create_task(acquire_lock_keeping_cancellation(lock))
    try:
        for _ in range(2):
            while not lock._waiters:
                await asyncio.sleep(0)
            acquiring.cancel()
            await asyncio.sleep(0)
            assert not acquiring.done()
    finally:
        lock.release()
    cancellation = await asyncio.wait_for(acquiring, 5)
    assert isinstance(cancellation, asyncio.CancelledError)
    assert lock.locked(), "the waiter does not hold the lock it was granted"
    lock.release()


@pytest.mark.asyncio
async def test_acquire_lock_keeping_cancellation_without_cancellation_returns_none():
    lock = asyncio.Lock()
    assert await acquire_lock_keeping_cancellation(lock) is None
    assert lock.locked()
    lock.release()
