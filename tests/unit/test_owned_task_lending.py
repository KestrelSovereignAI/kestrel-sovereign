"""Exact host-resource lending through the canonical task gate."""

import asyncio
import contextvars

import pytest

from kestrel_sovereign import async_ownership as ownership
from kestrel_sovereign._async_ownership import OwnedCallbackScope, without_owned_coordinators
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionStreamLifecycle,
    ProcessRuntimeExecutionFence,
    SourceWithdrawal,
    bind_execution_cleanup,
    bind_execution_custody,
    execution_stream,
    execution_evidence_carriers,
    owned_execution_stream,
    require_execution_work,
)
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


@pytest.mark.asyncio
async def test_exact_cleanup_does_not_request_a_new_work_loan(tmp_path):
    host = CapacityLoan(capacity=1)
    closed = asyncio.Event()
    path = tmp_path / "original-resource"
    resource = path.open("wb")

    async def body():
        capture = ownership.capture_owned_task_lender()
        assert capture is not None and len(host.outstanding) == 1

        async def release():
            assert ownership.capture_owned_task_lender() is None
            with pytest.raises(ExecutionAuthorityError, match="cleanup"):
                require_execution_work()
            resource.close()
            closed.set()

        with bind_execution_cleanup(None):
            await ownership.create_owned_task(release(), lender_capture=capture)
        assert closed.is_set() and len(host.outstanding) == 1

    try:
        with bind_execution_custody(ProcessRuntimeExecutionFence("owned-task-lender-test")):
            with ownership.bind_owned_task_lender(host):
                await ownership.create_owned_task(body())
    finally:
        resource.close()
    assert len(host.retired) == 1 and not host.outstanding


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign_reader", [False, True])
@pytest.mark.parametrize("lifecycle_factory", [False, True])
async def test_execution_stream_carries_original_work_loan_without_charging_its_closers(foreign_reader, lifecycle_factory):
    host = CapacityLoan(capacity=1)
    entered = []
    closed = []
    lifecycle = (ExecutionStreamLifecycle(ReentrantTransitionLock().capture_native_work_factory(), None, lambda: None)
                 if lifecycle_factory else None)

    async def source():
        producer = asyncio.current_task()
        entered.append(producer)
        assert ownership.capture_owned_task_lender() is not None
        assert [entry[1] for entry in host.outstanding.values()] == [producer]
        try:
            yield "original stream output"
        finally:
            closed.append(asyncio.current_task())

    async def consume(stream):
        assert await anext(stream) == "original stream output"
        assert len(host.outstanding) == 1

    with bind_execution_custody(ProcessRuntimeExecutionFence("owned-stream-lender-test")):
        with ownership.bind_owned_task_lender(host):
            stream = execution_stream(None, source(), lifecycle=lifecycle)
            try:
                if foreign_reader:
                    await asyncio.create_task(consume(stream))
                else:
                    await consume(stream)
            finally:
                await stream.aclose()
    assert entered == closed and len(entered) == 1
    assert not host.outstanding and host.retired[0].task is entered[0]


@pytest.mark.asyncio
async def test_ambiguous_exact_wrapper_never_receives_a_loan_or_dispatches():
    host = CapacityLoan()
    receipt = ownership.OwnedTaskAdmission()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    accepted, entered = [], []

    def duplicate(loop, coroutine, **options):
        accepted.extend(asyncio.Task(coroutine, loop=loop, **options) for _ in range(2))
        return accepted[0]

    async def body():
        entered.append(asyncio.current_task())

    with ownership.bind_owned_task_lender(host):
        loop.set_task_factory(duplicate)
        try:
            with pytest.raises(RuntimeError, match="original native operation admission failed"):
                ownership.create_owned_task(body(), admission=receipt)
        finally:
            loop.set_task_factory(previous)
            await asyncio.gather(*accepted, return_exceptions=True)
    assert not entered
    assert ownership.reconcile_owned_task_admission(receipt).state == "unknown"
    assert not host.retired and all(task is None for _, task, _ in host.outstanding.values())


@pytest.mark.asyncio
async def test_late_eager_wrapper_substitution_cannot_enter_the_payload():
    if not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("eager task factory unavailable")
    host = CapacityLoan()
    receipt = ownership.OwnedTaskAdmission()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    accepted, entered = [], []
    def factory(loop, coroutine, **options):
        def steal():
            accepted.append(asyncio.Task(coroutine, loop=loop, eager_start=True, **options))
        loop.call_soon(steal)
        return asyncio.Task(coroutine, loop=loop, **options)
    async def body():
        entered.append(asyncio.current_task())
    with ownership.bind_owned_task_lender(host):
        loop.set_task_factory(factory)
        try:
            task = ownership.create_owned_task(body(), admission=receipt)
        finally:
            loop.set_task_factory(previous)
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*accepted, return_exceptions=True)
    assert not entered
    assert ownership.reconcile_owned_task_admission(receipt).state == "unknown"
    assert not host.retired and len(host.outstanding) == 1


def test_foreign_loop_exact_wrapper_is_not_accepted_as_the_original_task():
    foreign = asyncio.new_event_loop()
    accepted, entered = [], []

    async def scenario():
        host = CapacityLoan()
        receipt = ownership.OwnedTaskAdmission()
        loop = asyncio.get_running_loop()
        previous = loop.get_task_factory()

        def wrong_loop(_loop, coroutine, **options):
            task = asyncio.Task(coroutine, loop=foreign, **options)
            accepted.append(task)
            return task

        async def body():
            entered.append(asyncio.current_task())

        with ownership.bind_owned_task_lender(host):
            loop.set_task_factory(wrong_loop)
            try:
                with pytest.raises(RuntimeError, match="original native operation admission failed"):
                    ownership.create_owned_task(body(), admission=receipt)
            finally:
                loop.set_task_factory(previous)
        assert ownership.reconcile_owned_task_admission(receipt).state == "unknown"
        assert all(task is None for _, task, _ in host.outstanding.values())

    try:
        asyncio.run(scenario())
    finally:
        if accepted:
            async def join_originals():
                await asyncio.gather(*accepted, return_exceptions=True)
            foreign.run_until_complete(join_originals())
        foreign.close()
    assert not entered


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["detached", "restored_cleanup", "withdrawn_stream"])
async def test_all_restrictions_suppress_even_explicit_old_lender_capture(boundary):
    from contextlib import ExitStack
    from kestrel_sovereign.execution_custody import (
        _StreamCleanupState, _bind_cleanup_flag, _bind_stream_cleanup,
    )

    host = CapacityLoan()
    fence = ProcessRuntimeExecutionFence("restricted-lender")
    async def independent():
        assert ownership.capture_owned_task_lender() is None
        assert not host.outstanding

    with ownership.bind_owned_task_lender(host):
        capture = ownership.capture_owned_task_lender()
        with ExitStack() as scope:
            if boundary == "detached":
                scope.enter_context(without_owned_coordinators())
            elif boundary == "restored_cleanup":
                scope.enter_context(_bind_cleanup_flag(True))
            else:
                scope.enter_context(_bind_stream_cleanup((_StreamCleanupState((fence,), True),)))
            await ownership.create_owned_task(independent(), lender_capture=capture)
            if boundary != "detached":
                with pytest.raises(RuntimeError, match="cleanup"):
                    with ownership.bind_owned_task_lender(CapacityLoan()):
                        pytest.fail("cleanup rebound WORK authority")
    assert not host.retired


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [False, True])
async def test_stream_paths_use_original_producer_receipt_and_captured_root(factory):
    from kestrel_sovereign._async_ownership import OwnedSourceAdmission, create_owned_iterator

    host = CapacityLoan(capacity=1)
    receipt = OwnedSourceAdmission()
    lock = ReentrantTransitionLock()
    producer = []
    async def source():
        producer.append(asyncio.current_task())
        assert [entry[1:] for entry in host.outstanding.values()] == [
            (asyncio.current_task(), receipt.producer),
        ]
        assert ownership.capture_owned_task_lender() is not None
        yield "same original reader"

    with ownership.bind_owned_task_lender(host):
        iterator = await create_owned_iterator(
            source, operation="lender-direct-stream", source_admission=receipt,
            owner_factory=lock.capture_native_work_factory() if factory else None,
        )
        try:
            assert iterator.producer_task is receipt.producer.task
            assert await anext(iterator) == "same original reader"
        finally:
            await iterator.aclose()
    assert host.retired[0].admission is receipt.producer and not host.outstanding


@pytest.mark.asyncio
async def test_eager_gate_withdrawal_does_not_lend_or_escape_admission_settlement():
    if not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("eager task factory unavailable")
    host = CapacityLoan()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    receipt = ownership.OwnedTaskAdmission()
    async def body():
        pytest.fail("withdrawn gate dispatched work")

    def factory(loop, coroutine, **options):
        task = asyncio.Task(coroutine, loop=loop, eager_start=True, **options)
        task.cancel("original factory cancellation")
        return task

    with ownership.bind_owned_task_lender(host):
        loop.set_task_factory(factory)
        try:
            with pytest.raises(RuntimeError, match="start gate"):
                ownership.create_owned_task(body(), admission=receipt)
            assert all(task is None for _, task, _ in host.outstanding.values())
        finally:
            loop.set_task_factory(previous)
            if receipt.task is not None:
                await asyncio.gather(receipt.task, return_exceptions=True)
    assert not host.outstanding and host.retired
    assert any(isinstance(error, asyncio.CancelledError)
               and error.args == ("original factory cancellation",)
               for error in execution_evidence_carriers(host.retired[0].error))


@pytest.mark.asyncio
async def test_stream_capacity_is_checked_before_any_structural_allocation():
    host = CapacityLoan(capacity=0)
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    events = []
    original_reserve = host.reserve
    def reserve(**options):
        events.append("reserve")
        return original_reserve(**options)
    host.reserve = reserve
    def factory(loop, coroutine, **options):
        events.append("allocate")
        return asyncio.Task(coroutine, loop=loop, **options)
    async def source():
        pytest.fail("capacity-refused producer dispatched")
        yield

    with ownership.bind_owned_task_lender(host):
        stream = execution_stream(None, source())
        loop.set_task_factory(factory)
        try:
            with pytest.raises(RuntimeError, match="test host capacity exhausted"):
                await anext(stream)
        finally:
            await stream.aclose()
            loop.set_task_factory(previous)
    assert events[0] == "reserve", events


@pytest.mark.asyncio
async def test_cooperative_stream_idle_waits_do_not_request_work_loans():
    host = CapacityLoan(capacity=1)
    async def source():
        assert ownership.capture_owned_task_lender() is not None
        assert len(host.outstanding) == 1
        yield "bounded cooperative stream"

    with ownership.bind_owned_task_lender(host):
        async with owned_execution_stream(None, source(), withdrawal=SourceWithdrawal()) as stream:
            await stream.reserve()
            assert await anext(stream) == "bounded cooperative stream"
    assert not host.outstanding and len(host.retired) == 1
