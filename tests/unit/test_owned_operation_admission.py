"""Finite observers do not replace or retire an operation's original admission."""

from __future__ import annotations

import asyncio

import pytest
from kestrel_sdk.resources import ResourceOutcomeUnknown

from kestrel_sovereign._async_ownership import (
    OwnedOperation,
    OwnedTaskAdmission,
    await_owned_task,
    create_owned_task,
)
from kestrel_sovereign import async_ownership as public
from kestrel_sovereign.execution_custody import execution_cleanup_debts, execution_evidence_carriers


def _refusing_completion_task_type(refusal):
    class RefuseOriginalCompletion(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == "close_unstarted":
                raise refusal
            return super().add_done_callback(callback, context=context)

    return RefuseOriginalCompletion


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
@pytest.mark.parametrize("borrowed", [True, False])
async def test_failed_opening_preserves_borrowed_observer_withdrawal(borrowed):
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    finalizer_entered, release_finalizer = asyncio.Event(), asyncio.Event()
    refusal = RuntimeError("original completion publication refused")
    business_entered = []
    accepted = []

    RefuseOriginalCompletion = _refusing_completion_task_type(refusal)

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

        observer = asyncio.create_task(operation.observe(
            defer_cancellation=not borrowed,
            cancel_on_caller_cancellation=not borrowed,
        ))
        await asyncio.sleep(0)
        observer.cancel("finite observer withdrew")
        # A test hang bound, not a resource-retirement deadline. The original
        # finalizer must remain blocked even after the borrowed observer leaves.
        done, _ = await asyncio.wait({observer}, timeout=0.2)
        if borrowed:
            assert observer in done, "failed opening silently changed a borrowed observer to must-join"
            observed = await await_owned_task(observer)
            assert isinstance(observed.error, asyncio.CancelledError)
            assert observed.error.args == ("finite observer withdrew",)
        else:
            assert observer not in done, "failed opening forwarded withdrawal into original finalizer"
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


@pytest.mark.asyncio
@pytest.mark.parametrize("with_resource", [False, True])
@pytest.mark.parametrize("body_fails", [False, True])
async def test_factory_publication_failure_is_on_original_retirement_receipt(with_resource, body_fails):
    admission = OwnedTaskAdmission()
    resource = object() if with_resource else None
    publication = RuntimeError("accepted owner's factory failed publication")
    body_error = LookupError("original accepted body failed") if body_fails else None

    async def original():
        if body_error is not None:
            raise body_error

    def factory(operation, context, receipt, retirement_resource):
        create_owned_task(operation, context=context, admission=receipt,
                          retirement_resource=retirement_resource)
        raise publication

    operation = OwnedOperation(original(), admission=admission,
                               retirement_resource=resource, owner_factory=factory)
    assert operation.task is admission.task
    await await_owned_task(operation.task)
    retirement = public.reconcile_owned_task_admission(admission)
    carriers = execution_evidence_carriers(retirement.error)
    assert publication in carriers
    if body_error is not None:
        assert body_error in carriers
    assert retirement.state == ("unknown" if with_resource else "complete")
    if with_resource:
        assert (resource, operation.task) in execution_cleanup_debts(retirement.error)


@pytest.mark.asyncio
async def test_nonacceptance_keeps_original_unstarted_resource_debt():
    admission, resource = OwnedTaskAdmission(), object()
    foreign_loop = asyncio.new_event_loop()

    async def original():
        raise AssertionError("refused native closer entered")

    payload = original()
    try:
        with pytest.raises(RuntimeError) as caught:
            create_owned_task(payload, admission=admission, loop=foreign_loop,
                              retirement_resource=resource)
        assert payload.cr_frame is None
        retirement = public.reconcile_owned_task_admission(admission)
        assert retirement.state == "nonacceptance"
        assert retirement.error is caught.value
        assert execution_cleanup_debts(retirement.error) == ((resource, None),)
    finally:
        payload.close()
        foreign_loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("early_cancel", [False, True])
async def test_reconciliation_keeps_explicit_receipt_when_task_metadata_is_denied(early_cancel):
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    receipt, resource = OwnedTaskAdmission(), object()
    entered = []

    class DenyMetadata(asyncio.Task):
        def __setattr__(self, name, value):
            if name in {"_kestrel_owned_admission", "_kestrel_owned_terminal_error"}:
                raise RuntimeError("diagnostic Task metadata publication denied")
            super().__setattr__(name, value)

    def factory(loop, operation, **kwargs):
        return DenyMetadata(operation, loop=loop, **kwargs)

    async def original():
        entered.append(asyncio.current_task())

    operation = None
    try:
        loop.set_task_factory(factory)
        operation = OwnedOperation(original(), admission=receipt, retirement_resource=resource)
        loop.set_task_factory(previous)
        assert operation.task is receipt.task
        if early_cancel:
            operation.task.cancel("before original first step")
        await await_owned_task(operation.task)
        snapshot = public.reconcile_owned_task_admission(receipt)
        assert snapshot.state == "unknown"
        assert (resource, operation.task) in execution_cleanup_debts(snapshot.error)
        assert not entered
    finally:
        loop.set_task_factory(previous)
        if operation is not None and operation.task is not None:
            await await_owned_task(operation.task)


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("reconcile_first", [False, True])
async def test_operation_observers_keep_original_receipt_after_metadata_refusal(borrowed, reconcile_first):
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    receipt = OwnedTaskAdmission()
    unknown = ResourceOutcomeUnknown("original physical finalizer outcome uncertain")
    physical = asyncio.CancelledError("original physical finalizer")
    physical.__cause__ = unknown

    class RefuseMetadata(asyncio.Task):
        def __setattr__(self, name, value):
            if name in {"_kestrel_owned_admission", "_kestrel_owned_terminal_error"}:
                raise RuntimeError("original native metadata publication refused")
            super().__setattr__(name, value)

    def factory(loop, operation, **kwargs):
        async def original_finalizer():
            try:
                return await operation
            finally:
                raise physical

        return RefuseMetadata(original_finalizer(), loop=loop, **kwargs)

    async def body():
        pytest.fail("metadata publication failed before payload admission")

    loop.set_task_factory(factory)
    try:
        operation = OwnedOperation(body(), admission=receipt)
    finally:
        loop.set_task_factory(previous)
    assert operation.task is receipt.task and operation.task is not None

    # Native wait gives us terminalization without consuming original C.
    await asyncio.wait((operation.task,))
    if reconcile_first:
        retirement = public.reconcile_owned_task_admission(receipt)
        assert unknown in execution_evidence_carriers(retirement.error)
    seen = await operation.observe(defer_cancellation=not borrowed)
    assert seen.cancellation is None
    assert unknown in execution_evidence_carriers(seen.error)
    assert unknown in execution_evidence_carriers(public.reconcile_owned_task_admission(receipt).error)


@pytest.mark.asyncio
async def test_returning_failed_gated_factory_retains_same_resource_receipt():
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    receipt, resource = OwnedTaskAdmission(), object()
    refusal = RuntimeError("original completion subscription refused")
    task_type = _refusing_completion_task_type(refusal)

    async def original():
        raise AssertionError("failed gate dispatched business body")

    def native_factory(loop, operation, **kwargs):
        return task_type(operation, loop=loop, **kwargs)

    def owner_factory(operation, context, admission, retirement_resource):
        try:
            return create_owned_task(operation, admission=admission,
                                     retirement_resource=retirement_resource)
        except RuntimeError:
            return admission.task

    operation = None
    try:
        loop.set_task_factory(native_factory)
        operation = OwnedOperation(original(), admission=receipt,
                                   retirement_resource=resource, owner_factory=owner_factory)
        loop.set_task_factory(previous)
        await await_owned_task(operation.task)
        snapshot = public.reconcile_owned_task_admission(receipt)
        assert snapshot.state == "unknown"
        assert (resource, operation.task) in execution_cleanup_debts(snapshot.error)
        assert refusal in execution_evidence_carriers(snapshot.error)
    finally:
        loop.set_task_factory(previous)
        if operation is not None and operation.task is not None:
            await await_owned_task(operation.task)


@pytest.mark.asyncio
async def test_factory_refusal_before_native_creation_consumes_receipt():
    receipt = OwnedTaskAdmission()
    refusal = RuntimeError("owner factory refused before native creation")

    async def original():
        raise AssertionError("refusing factory dispatched work")

    def factory(operation, context, admission, retirement_resource):
        operation.close()
        raise refusal

    first = OwnedOperation(original(), admission=receipt, owner_factory=factory)
    assert first.task is None
    second_payload = original()
    second = None
    try:
        with pytest.raises(TypeError, match="fresh original receipt"):
            second = OwnedOperation(second_payload, admission=receipt)
        assert second_payload.cr_frame is None
    finally:
        if second is not None and second.task is not None:
            await await_owned_task(second.task)
        else:
            second_payload.close()


@pytest.mark.asyncio
async def test_constructor_claim_precedes_reentrant_factory_dispatch():
    receipt = OwnedTaskAdmission()
    nested = []
    refused = []

    async def original():
        return "only an original, separately admitted body may execute"

    def factory(operation, context, admission, retirement_resource):
        operation.close()
        try:
            nested.append(OwnedOperation(original(), admission=receipt))
        except TypeError:
            refused.append(True)
        raise RuntimeError("outer factory refused")

    outer = OwnedOperation(original(), admission=receipt, owner_factory=factory)
    try:
        assert refused == [True] and not nested
        assert outer.task is None
    finally:
        for operation in nested:
            if operation.task is not None:
                await await_owned_task(operation.task)
