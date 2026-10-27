"""Public aliases retain canonical task ownership, not execution authority."""

from __future__ import annotations

import asyncio
import inspect
import sys
import threading

import pytest

from kestrel_sovereign import _async_ownership as canonical
from kestrel_sovereign import async_ownership as public
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    ProcessRuntimeExecutionFence,
    bind_execution_custody,
    execution_cleanup_debts,
    execution_denial_scopes,
    execution_commit_outcome,
    execution_observation_error,
    execution_terminal_error,
    run_execution_blocking_operation,
)


def test_public_owned_tail_api_is_exact_canonical_identity_and_signature():
    names = {
        "OwnedTaskAdmission",
        "OwnedTaskOutcome",
        "await_owned_operation",
        "await_owned_task",
        "create_owned_task",
        "owned_task_error",
        "pending_owned_retirement_error",
        "raise_owned_outcome",
        "run_blocking_operation",
        "wait_owned_tasks_until",
    }
    assert set(public.__all__) == names
    for name in names:
        alias = getattr(public, name)
        original = getattr(canonical, name)
        assert alias is original
        assert inspect.signature(alias) == inspect.signature(original)


@pytest.mark.asyncio
@pytest.mark.parametrize("control", [False, True])
async def test_public_operation_retains_original_outcome_and_cleanup(control):
    original = (
        ExecutionAuthorityError("original work denial")
        if control
        else LookupError("original cleanup failure")
    )
    cleanup = []

    async def retire():
        try:
            raise original
        finally:
            cleanup.append(asyncio.current_task())

    outcome = await public.await_owned_operation(retire(), retain_originals=True)

    assert type(outcome) is public.OwnedTaskOutcome
    assert outcome.error is original
    assert outcome.cancellation is None
    assert len(cleanup) == 1 and cleanup[0].done()
    with pytest.raises(type(original)) as caught:
        public.raise_owned_outcome(
            outcome, operation="original retirement", retain_originals=True
        )
    assert caught.value is original


async def _wait_for_thread_phase(phase: threading.Event) -> None:
    # A test hang bound, not a runtime cleanup deadline or a work permission.
    async with asyncio.timeout(5):
        while not phase.is_set():
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_public_blocking_tail_joins_body_and_finalizer_before_withdrawal():
    body_entered = threading.Event()
    finish_body = threading.Event()
    finalizer_entered = threading.Event()
    finish_finalizer = threading.Event()
    physically_closed = threading.Event()
    threads = []
    caller_thread = threading.get_ident()

    def original_tail():
        threads.append(threading.get_ident())
        try:
            body_entered.set()
            finish_body.wait()
        finally:
            finalizer_entered.set()
            finish_finalizer.wait()
            physically_closed.set()

    owner = asyncio.create_task(public.run_blocking_operation(original_tail))
    try:
        await _wait_for_thread_phase(body_entered)
        owner.cancel("withdrawal during original body")
        await asyncio.sleep(0)
        assert not owner.done() and not finalizer_entered.is_set()

        finish_body.set()
        await _wait_for_thread_phase(finalizer_entered)
        owner.cancel("withdrawal during original finalizer")
        await asyncio.sleep(0)
        assert not owner.done() and not physically_closed.is_set()

        finish_finalizer.set()
        outcome = await public.await_owned_task(owner)
        assert physically_closed.is_set()
        assert threads and threads[0] != caller_thread
        assert isinstance(outcome.error, asyncio.CancelledError)
        assert outcome.error.args == ("withdrawal during original body",)
        with pytest.raises(asyncio.CancelledError) as caught:
            public.raise_owned_outcome(
                outcome, operation="original blocking retirement", retain_originals=True
            )
        assert caught.value is outcome.error
    finally:
        finish_body.set()
        finish_finalizer.set()
        await public.await_owned_task(owner)


@pytest.mark.asyncio
async def test_public_cleanup_joins_original_tail_after_work_revocation():
    body_entered = threading.Event()
    finish_body = threading.Event()
    physically_closed = threading.Event()
    fresh_calls = []

    def original_tail():
        try:
            body_entered.set()
            finish_body.wait()
        finally:
            physically_closed.set()

    def fresh_work():
        fresh_calls.append("unexpected admission")

    fence = ProcessRuntimeExecutionFence("did:test:public-owned-tail")
    with bind_execution_custody(fence) as original_scope:
        # Real process-lifetime WORK checks run before admission and in the
        # physical thread. This is not distributed tenant/generation proof.
        owner = asyncio.create_task(
            run_execution_blocking_operation(None, original_tail)
        )
        cleanup = None
        try:
            await _wait_for_thread_phase(body_entered)
            fence.retire()
            with pytest.raises(ExecutionAuthorityError) as refused:
                await run_execution_blocking_operation(None, fresh_work)
            assert execution_denial_scopes(refused.value) == (original_scope,)
            assert fresh_calls == []

            # Joining already-owned work requires no replacement admission.
            cleanup = asyncio.create_task(public.await_owned_task(owner))
            await asyncio.sleep(0)
            assert not cleanup.done() and not physically_closed.is_set()
            finish_body.set()
            outcome = await cleanup
            assert owner.done() and physically_closed.is_set()
            assert isinstance(outcome.error, ExecutionAuthorityError)
            assert execution_denial_scopes(outcome.error) == (original_scope,)
            with pytest.raises(ExecutionAuthorityError) as terminal:
                public.raise_owned_outcome(
                    outcome, operation="revoked original tail", retain_originals=True
                )
            assert terminal.value is outcome.error
            with pytest.raises(ExecutionAuthorityError):
                await run_execution_blocking_operation(None, fresh_work)
            assert fresh_calls == []
        finally:
            finish_body.set()
            await public.await_owned_task(owner)
            if cleanup is not None:
                await public.await_owned_task(cleanup)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["allocation", "callback-before", "callback-after"])
async def test_public_admission_keeps_exact_native_task_across_publication_failure(failure):
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    receipt = public.OwnedTaskAdmission()
    refusal = RuntimeError("original public admission refusal: " + failure)
    accepted = []
    entered = []

    async def original():
        entered.append(asyncio.current_task())

    operation = original()

    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == "close_unstarted":
                assert receipt.task is self
                assert not entered
                if failure == "callback-after":
                    super().add_done_callback(callback, context=context)
                raise refusal
            return super().add_done_callback(callback, context=context)

    def factory(loop, coroutine, **kwargs):
        if failure == "allocation":
            raise refusal
        task = PublicationTask(coroutine, loop=loop, **kwargs)
        accepted.append(task)
        return task

    try:
        # This factory is installed only across this synchronous admission.
        loop.set_task_factory(factory)
        with pytest.raises(RuntimeError) as caught:
            public.create_owned_task(operation, admission=receipt, retain_originals=True)
        if failure == "allocation":
            from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
            assert refusal in _explicit_control_chain(caught.value)
            assert receipt._acceptance_unknown
            assert execution_cleanup_debts(caught.value)
        else:
            assert caught.value is refusal
    finally:
        loop.set_task_factory(previous_factory)
        primary = sys.exception()
        joined = None
        if receipt.task is not None:
            joined = await public.await_owned_task(receipt.task)
        terminal = execution_observation_error(
            primary,
            None if joined is None or joined.error is refusal else joined.error,
            None if joined is None else joined.cancellation,
        )
        if terminal is not None:
            raise terminal

    assert inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED
    assert entered == []
    if failure == "allocation":
        assert receipt.task is None and accepted == []
    else:
        assert accepted == [receipt.task]
        assert receipt.task.done() and not receipt.task.cancelled()
        assert joined.error is refusal
        assert public.owned_task_error(receipt.task) is refusal
        assert public.owned_task_error(receipt.task) is refusal


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["ordinary", "control", "unknown", "cancellation"])
async def test_public_expired_deadline_retains_error_and_pending_original_debt(failure):
    errors = {
        "ordinary": LookupError("original producer error"),
        "control": ExecutionAuthorityError("original producer denial"),
        "unknown": ExecutionCommitOutcomeError("unknown"),
        "cancellation": asyncio.CancelledError("original producer cancellation"),
    }
    original_error = errors[failure]
    release = asyncio.Event()
    entered = asyncio.Event()
    retired = []
    resource = object()

    async def fail():
        raise original_error

    async def pending():
        entered.set()
        try:
            await release.wait()
        finally:
            retired.append(asyncio.current_task())

    failed_owner = public.create_owned_task(fail(), retain_originals=True)
    pending_owner = public.create_owned_task(pending(), retain_originals=True)
    try:
        async with asyncio.timeout(5):
            await entered.wait()
        failed = await public.await_owned_task(failed_owner)
        assert failed.error is original_error
        assert public.owned_task_error(failed_owner) is original_error
        assert public.owned_task_error(failed_owner) is original_error
        with pytest.raises(RuntimeError, match="has not reached terminal state"):
            public.owned_task_error(pending_owner)

        observed = await public.wait_owned_tasks_until(
            (failed_owner, pending_owner), deadline=asyncio.get_running_loop().time()
        )
        assert observed.result == ({failed_owner}, {pending_owner})
        assert observed.error is None and observed.cancellation is None
        assert not pending_owner.done() and not pending_owner.cancelling()
        assert retired == []
        debt = public.pending_owned_retirement_error(
            resource, (pending_owner, pending_owner, failed_owner),
            message="original public retirement remains pending",
        )
        assert type(debt) is TimeoutError
        assert execution_cleanup_debts(debt) == ((resource, pending_owner),)
        terminal = execution_terminal_error(original_error, debt, retain_originals=True)
        assert execution_cleanup_debts(terminal) == ((resource, pending_owner),)
        assert original_error is terminal or original_error is terminal.__cause__
        if failure == "unknown":
            # A genuine typed unit receipt, not an observed database COMMIT.
            assert execution_commit_outcome(terminal) == "unknown"
    finally:
        primary = sys.exception()
        release.set()
        failed = await public.await_owned_task(failed_owner)
        pending_outcome = await public.await_owned_task(pending_owner)
        terminal = execution_observation_error(
            primary,
            None if failed.error is original_error else failed.error,
            failed.cancellation, pending_outcome.error, pending_outcome.cancellation,
        )
        if terminal is not None:
            raise terminal
    assert retired == [pending_owner]
    assert public.pending_owned_retirement_error(
        resource, (pending_owner,), message="already retired original"
    ) is None
    # Completing the owner does not rewrite an earlier finite debt snapshot.
    assert execution_cleanup_debts(debt) == ((resource, pending_owner),)


@pytest.mark.asyncio
async def test_public_bounded_wait_retains_repeated_observer_cancellation_without_cancelling_owner():
    entered = asyncio.Event()
    release = asyncio.Event()
    retired = []

    async def original():
        try:
            await release.wait()
            return "same original result"
        finally:
            retired.append(asyncio.current_task())

    owner = public.create_owned_task(original(), retain_originals=True)
    # A test hang bound, not a production work/retirement SLA.
    deadline = asyncio.get_running_loop().time() + 5

    async def observe():
        entered.set()
        return await public.wait_owned_tasks_until((owner,), deadline=deadline)

    observer = public.create_owned_task(observe(), retain_originals=True)
    try:
        await entered.wait()
        observer.cancel("first observer withdrawal")
        await asyncio.sleep(0)
        assert not observer.done() and not owner.done() and not owner.cancelling()
        observer.cancel("second observer withdrawal")
        await asyncio.sleep(0)
        assert not observer.done() and not owner.done() and not owner.cancelling()
        release.set()
        joined = await public.await_owned_task(observer)
        assert joined.error is None and joined.cancellation is None
        observed = joined.result
        assert observed.result == ({owner}, set()) and observed.error is None
        assert isinstance(observed.cancellation, asyncio.CancelledError)
        assert observed.cancellation.args == ("first observer withdrawal",)
        assert isinstance(observed.cancellation.__cause__, asyncio.CancelledError)
        assert observed.cancellation.__cause__.args == ("second observer withdrawal",)
        assert retired == [owner]
        with pytest.raises(asyncio.CancelledError) as caught:
            public.raise_owned_outcome(observed, operation="bounded public observation", retain_originals=True)
        assert caught.value is observed.cancellation
    finally:
        primary = sys.exception()
        release.set()
        joined = await public.await_owned_task(observer)
        original_outcome = await public.await_owned_task(owner)
        terminal = execution_observation_error(
            primary, joined.error, joined.cancellation,
            original_outcome.error, original_outcome.cancellation,
        )
        if terminal is not None:
            raise terminal
