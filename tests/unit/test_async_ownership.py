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
@pytest.mark.parametrize("allocation", ["completion", "shield"])
async def test_completion_allocation_refusal_cannot_abandon_original(monkeypatch, allocation):
    release, refused = asyncio.Event(), asyncio.Event()
    failure = RuntimeError("controlled original completion allocation refusal")

    async def original():
        await release.wait()
        return "original result"

    source = create_owned_task(original(), retain_originals=True)
    loop = asyncio.get_running_loop()
    allocate = loop.create_future

    def controlled_allocation():
        frame = sys._getframe(1)
        if allocation == "shield" and frame.f_code.co_name == "shield":
            frame = frame.f_back
        if (not refused.is_set() and frame.f_code.co_name == "await_owned_task"
                and frame.f_locals.get("task") is source
                and (allocation == "completion" or sys._getframe(1).f_code.co_name == "shield")):
            refused.set()
            raise failure
        return allocate()

    monkeypatch.setattr(loop, "create_future", controlled_allocation)
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
async def test_cancelling_unstarted_iterator_owner_interrupts_source():
    """The exposed cancellation owner works before its coroutine first runs."""

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
    # This is the endpoint race: registration exposes the watcher and Stop
    # cancels it in the same event-loop turn, before its coroutine body runs.
    assert owned.owner_task.cancel() is True

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
