"""Public aliases retain canonical task ownership, not execution authority."""

from __future__ import annotations

import asyncio
import inspect
import threading

import pytest

from kestrel_sovereign import _async_ownership as canonical
from kestrel_sovereign import async_ownership as public
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ProcessRuntimeExecutionFence,
    bind_execution_custody,
    execution_denial_scopes,
    run_execution_blocking_operation,
)


def test_public_owned_tail_api_is_exact_canonical_identity_and_signature():
    names = {
        "OwnedTaskOutcome",
        "await_owned_operation",
        "await_owned_task",
        "raise_owned_outcome",
        "run_blocking_operation",
    }
    assert set(public.__all__) == names
    for name in names:
        alias = getattr(public, name)
        original = getattr(canonical, name)
        assert alias is original
        assert inspect.signature(alias) == inspect.signature(original)
    assert not hasattr(public, "create_owned_task")
    assert not hasattr(public, "OwnedTaskAdmission")


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
