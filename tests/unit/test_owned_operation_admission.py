"""Finite observers do not replace or retire an operation's original admission."""

from __future__ import annotations

import asyncio

import pytest

from kestrel_sovereign._async_ownership import (
    OwnedOperation,
    OwnedTaskAdmission,
    await_owned_task,
    create_owned_task,
)
from kestrel_sovereign import async_ownership as public


@pytest.mark.asyncio
@pytest.mark.parametrize("use_factory", [False, True])
async def test_operation_uses_callers_preallocated_original_admission(use_factory):
    admission = OwnedTaskAdmission()
    entered = []
    factory_receipts = []

    async def original():
        entered.append(asyncio.current_task())
        return "original result"

    def factory(operation, context, receipt, retirement_resource):
        factory_receipts.append(receipt)
        return create_owned_task(
            operation, context=context, admission=receipt,
            retirement_resource=retirement_resource,
        )

    payload = original()
    operation = None
    try:
        operation = OwnedOperation(
            payload, admission=admission,
            owner_factory=factory if use_factory else None,
        )
        assert admission.task is operation.task
        outcome = await operation.observe()
        assert outcome.result == "original result" and outcome.error is None
        assert entered == [admission.task]
        assert factory_receipts == ([admission] if use_factory else [])
    finally:
        # A construction refusal must not leak the still-unstarted payload.
        if operation is None:
            payload.close()
        elif operation.task is not None:
            await await_owned_task(operation.task)


@pytest.mark.asyncio
async def test_failed_opening_preserves_borrowed_observer_withdrawal():
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    finalizer_entered, release_finalizer = asyncio.Event(), asyncio.Event()
    refusal = RuntimeError("original completion publication refused")
    business_entered = []
    accepted = []

    class RefuseOriginalCompletion(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == "close_unstarted":
                raise refusal
            return super().add_done_callback(callback, context=context)

    def factory(loop, wrapper, **kwargs):
        async def outer():
            try:
                return await wrapper
            finally:
                finalizer_entered.set()
                await release_finalizer.wait()

        task = RefuseOriginalCompletion(outer(), loop=loop, **kwargs)
        accepted.append(task)
        return task

    async def original():
        business_entered.append(asyncio.current_task())

    observer = None
    operation = None
    try:
        loop.set_task_factory(factory)
        operation = OwnedOperation(original(), retain_originals=True)
        loop.set_task_factory(previous)
        assert operation.task is accepted[0]
        assert operation.admission_error is not None
        await finalizer_entered.wait()
        assert not business_entered and not operation.task.done()

        observer = asyncio.create_task(operation.observe(defer_cancellation=False))
        await asyncio.sleep(0)
        observer.cancel("finite observer withdrew")
        # A test hang bound, not a resource-retirement deadline. The original
        # finalizer must remain blocked even after the borrowed observer leaves.
        done, _ = await asyncio.wait({observer}, timeout=0.2)
        assert observer in done, "failed opening silently changed a borrowed observer to must-join"
        observed = await await_owned_task(observer)
        assert isinstance(observed.error, asyncio.CancelledError)
        assert observed.error.args == ("finite observer withdrew",)
        assert not operation.task.done() and not release_finalizer.is_set()
    finally:
        loop.set_task_factory(previous)
        release_finalizer.set()
        if operation is not None and operation.task is not None:
            await await_owned_task(operation.task)
        if observer is not None:
            await await_owned_task(observer)


def test_fresh_receipt_is_not_proven_nonacceptance():
    admission = OwnedTaskAdmission()
    retirement = public.reconcile_owned_task_admission(admission)
    assert retirement.admission is admission and retirement.task is None
    assert retirement.state == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_reconciliation_uses_exact_original_physical_completion(failure):
    entered, finish = asyncio.Event(), asyncio.Event()
    admission = OwnedTaskAdmission()
    original_error = LookupError("original body refusal") if failure else None

    async def original():
        entered.set()
        await finish.wait()
        if original_error is not None:
            raise original_error

    task = create_owned_task(original(), admission=admission)
    try:
        await entered.wait()
        pending = public.reconcile_owned_task_admission(admission)
        assert pending.state == "pending" and pending.task is task
        finish.set()
        outcome = await await_owned_task(task)
        final = public.reconcile_owned_task_admission(admission)
        assert final.state == "complete" and final.task is task
        assert final.admission is admission
        assert final.error is outcome.error is original_error
    finally:
        finish.set()
        await await_owned_task(task)


@pytest.mark.asyncio
async def test_proven_refusal_requires_original_unstarted_payload_close():
    admission = OwnedTaskAdmission()
    foreign_loop = asyncio.new_event_loop()

    async def original():
        raise AssertionError("business body entered after proven refusal")

    payload = original()
    try:
        with pytest.raises(RuntimeError, match="original running loop"):
            create_owned_task(payload, loop=foreign_loop, admission=admission)
        assert payload.cr_frame is None
        retirement = public.reconcile_owned_task_admission(admission)
        assert retirement.state == "nonacceptance"
        assert retirement.task is None and retirement.admission is admission
    finally:
        payload.close()
        foreign_loop.close()


@pytest.mark.asyncio
async def test_opaque_factory_refusal_never_proves_nonacceptance():
    admission = OwnedTaskAdmission()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    refusal = RuntimeError("opaque factory acceptance cannot be observed")

    async def original():
        raise AssertionError("business body entered after unknown acceptance")

    def factory(loop, wrapper, **kwargs):
        raise refusal

    try:
        loop.set_task_factory(factory)
        with pytest.raises(RuntimeError) as caught:
            create_owned_task(original(), admission=admission)
        assert caught.value is not None
    finally:
        loop.set_task_factory(previous)
    retirement = public.reconcile_owned_task_admission(admission)
    assert retirement.state == "unknown" and retirement.task is None
    assert retirement.error is caught.value


@pytest.mark.asyncio
@pytest.mark.parametrize("use_factory", [False, True])
async def test_operation_refuses_consumed_receipt_before_factory_dispatch(use_factory):
    admission = OwnedTaskAdmission()
    dispatched = []

    async def first():
        return "first"

    owner = create_owned_task(first(), admission=admission)
    await await_owned_task(owner)

    async def second():
        dispatched.append("unexpected business dispatch")

    def factory(operation, context, receipt, retirement_resource):
        dispatched.append("unexpected factory dispatch")
        return owner

    payload = second()
    try:
        with pytest.raises(TypeError, match="fresh original receipt"):
            OwnedOperation(payload, admission=admission,
                           owner_factory=factory if use_factory else None)
        assert not dispatched
        assert payload.cr_frame is None
        assert admission.task is owner
    finally:
        payload.close()


@pytest.mark.asyncio
async def test_completed_task_cannot_ack_its_unretired_physical_resource():
    admission = OwnedTaskAdmission()
    resource = object()
    debt = RuntimeError("original physical closer refused")
    debt._kestrel_native_retirement_debts = ((resource, None),)

    async def original():
        raise debt

    task = create_owned_task(original(), admission=admission, retain_originals=True)
    outcome = await await_owned_task(task)
    assert task.done()
    retirement = public.reconcile_owned_task_admission(admission)
    assert retirement.state == "unknown"
    assert retirement.error is outcome.error
