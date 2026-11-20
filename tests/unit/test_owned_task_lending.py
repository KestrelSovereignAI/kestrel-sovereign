"""Exact host-resource lending through the canonical task gate."""

import asyncio
import contextvars

import pytest

from kestrel_sovereign import async_ownership as ownership
from kestrel_sovereign._async_ownership import OwnedCallbackScope, without_owned_coordinators
from kestrel_sovereign.execution_custody import execution_evidence_carriers
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock


class CapacityLoan:
    """A bounded host resource with one accounting entry per original receipt."""

    def __init__(self, capacity=8):
        self.parent = asyncio.current_task()
        self.capacity = capacity
        self.outstanding = {}
        self.retired = []
        self.denial = None

    def reserve(self, *, creator, original_parent, admission, structural_role):
        assert original_parent is self.parent or any(
            task is original_parent for _, task, _ in self.outstanding.values()
        )
        assert admission.task is None
        if len(self.outstanding) >= self.capacity:
            raise RuntimeError("test host capacity exhausted")
        key = object()
        self.outstanding[key] = (creator, None, admission)
        return key

    def lend(self, reservation, *, task, admission):
        creator, _, original_admission = self.outstanding[reservation]
        assert creator is asyncio.current_task()
        assert admission is original_admission and admission.task is task
        self.outstanding[reservation] = (creator, task, admission)
        if self.denial is not None:
            raise self.denial

    def retire(self, reservation, *, retirement):
        _, task, admission = self.outstanding[reservation]
        assert retirement.admission is admission
        if retirement.state == "complete":
            assert retirement.task is admission.task and retirement.task.done()
            assert task is None or retirement.task is task
        else:
            assert retirement.state == "nonacceptance"
            assert task is None and retirement.task is None
        del self.outstanding[reservation]
        self.retired.append(retirement)


@pytest.mark.asyncio
async def test_lends_exact_task_before_payload_and_retains_physical_finalizer():
    host = CapacityLoan()
    release = asyncio.Event()
    entered = asyncio.Event()
    receipt = ownership.OwnedTaskAdmission()

    async def body():
        task = asyncio.current_task()
        assert list(host.outstanding.values()) == [(host.parent, task, receipt)]
        try:
            entered.set()
            return task
        finally:
            await release.wait()

    with ownership.bind_owned_task_lender(host):
        task = ownership.create_owned_task(body(), admission=receipt)
        try:
            await entered.wait()
            assert not host.retired
            assert ownership.reconcile_owned_task_admission(receipt).state == "pending"
        finally:
            release.set()
        assert await task is task
    assert not host.outstanding
    assert [item.admission for item in host.retired] == [receipt]


@pytest.mark.asyncio
async def test_explicit_fresh_context_lends_to_child_but_inherited_route_does_not():
    host = CapacityLoan()
    marker = contextvars.ContextVar("test_lender_private_marker", default=None)
    token = marker.set("original")

    async def fresh_body():
        assert marker.get() is None
        assert ownership.capture_owned_task_lender() is not None
        await ownership.create_owned_task(nested_body())

    async def nested_body():
        assert marker.get() is None
        assert ownership.capture_owned_task_lender() is not None

    async def raw_body():
        assert marker.get() == "original"
        assert ownership.capture_owned_task_lender() is None
        await ownership.create_owned_task(independent_body())

    async def independent_body():
        assert ownership.capture_owned_task_lender() is None
        assert not host.outstanding

    try:
        with ownership.bind_owned_task_lender(host):
            capture = ownership.capture_owned_task_lender()
            await asyncio.create_task(raw_body())
            await ownership.create_owned_task(independent_body(), context=contextvars.copy_context())
            await ownership.create_owned_task(
                fresh_body(), context=contextvars.Context(), lender_capture=capture,
            )
        with pytest.raises(RuntimeError, match="lender.*no longer live"):
            ownership.create_owned_task(independent_body(), lender_capture=capture)
    finally:
        marker.reset(token)
    assert not host.outstanding and len(host.retired) == 2


@pytest.mark.asyncio
async def test_capacity_refusal_precedes_any_native_allocation():
    host = CapacityLoan(capacity=0)
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    allocations = []

    def factory(loop, coroutine, **kwargs):
        allocations.append(coroutine)
        return asyncio.Task(coroutine, loop=loop, **kwargs)

    async def body():
        pytest.fail("refused payload entered")

    with ownership.bind_owned_task_lender(host):
        loop.set_task_factory(factory)
        try:
            with pytest.raises(RuntimeError, match="test host capacity exhausted"):
                ownership.create_owned_task(body())
        finally:
            loop.set_task_factory(previous)
    assert not allocations and not host.outstanding


@pytest.mark.asyncio
async def test_refused_lend_keeps_original_receipt_through_pre_entry_cancellation():
    host = CapacityLoan()
    host.denial = RuntimeError("test host vetoed accepted task")
    receipt = ownership.OwnedTaskAdmission()

    async def body():
        pytest.fail("failed start gate entered")

    with ownership.bind_owned_task_lender(host):
        with pytest.raises(RuntimeError, match="test host vetoed accepted task"):
            ownership.create_owned_task(body(), admission=receipt)
        task = receipt.task
        assert task is not None and not host.retired
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not host.outstanding and len(host.retired) == 1
    retired = host.retired[0]
    assert retired.task is task and retired.error is not None
    assert ownership.reconcile_owned_task_admission(receipt) == retired


@pytest.mark.asyncio
async def test_opaque_factory_keeps_unknown_loan_without_dispatching_payload():
    host = CapacityLoan()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    receipt = ownership.OwnedTaskAdmission()
    invoked = False

    def factory(loop, coroutine, **kwargs):
        async def opaque():
            return await coroutine

        return asyncio.Task(opaque(), loop=loop, **kwargs)

    async def body():
        nonlocal invoked
        invoked = True

    with ownership.bind_owned_task_lender(host):
        loop.set_task_factory(factory)
        try:
            with pytest.raises(RuntimeError, match="original native operation admission failed") as opening:
                ownership.create_owned_task(body(), admission=receipt)
            assert any("opaque native task" in str(error)
                       for error in execution_evidence_carriers(opening.value))
        finally:
            loop.set_task_factory(previous)
        assert receipt.task is not None
        with pytest.raises(RuntimeError, match="original native operation admission failed"):
            await receipt.task
    assert not invoked and not host.retired
    assert len(host.outstanding) == 1
    assert ownership.reconcile_owned_task_admission(receipt).state == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["callback", "privacy_factory"])
async def test_fresh_callback_uses_original_capture_not_actual_creators_inheritance(boundary):
    host = CapacityLoan()
    marker = contextvars.ContextVar("test_lender_callback_marker", default=None)
    token = marker.set("private original context")
    actual_creator = None

    async def body():
        assert marker.get() is None
        assert ownership.capture_owned_task_lender() is not None
        assert [entry[:2] for entry in host.outstanding.values()] == [
            (actual_creator, asyncio.current_task()),
        ]

    async def callback(start):
        nonlocal actual_creator
        actual_creator = asyncio.current_task()
        assert actual_creator is not host.parent
        assert ownership.capture_owned_task_lender() is None
        await start(body, context=contextvars.Context())

    try:
        with ownership.bind_owned_task_lender(host):
            start = (OwnedCallbackScope().start if boundary == "callback" else
                     ReentrantTransitionLock().capture_native_work_factory(acquire_span=False))
            await asyncio.create_task(callback(start))
    finally:
        marker.reset(token)
    assert not host.outstanding and len(host.retired) == 1


@pytest.mark.asyncio
async def test_factory_captured_outside_a_root_cannot_enroll_in_active_root():
    host = CapacityLoan()
    lock = ReentrantTransitionLock()
    start = lock.capture_native_work_factory(acquire_span=False)

    async def body():
        assert ownership.capture_owned_task_lender() is None
        assert not host.outstanding

    with ownership.bind_owned_task_lender(host):
        await start(body)
        with without_owned_coordinators():
            await ownership.create_owned_task(body())
    assert not host.retired
