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
    close_execution_stream,
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


@pytest.mark.parametrize("flavor", ["asyncio", "auto"])
@pytest.mark.parametrize("substitute", [False, True])
def test_actual_uvicorn_loop_selection_preserves_native_loan_ownership(flavor, substitute, monkeypatch):
    import importlib.util
    import uvicorn
    loop = uvicorn.Config("unused:app", loop=flavor, log_config=None).get_loop_factory()()
    if flavor == "auto" and importlib.util.find_spec("uvloop") is not None:
        import uvloop
        assert isinstance(loop, uvloop.Loop), "the installed production loop was not qualified"
    async def scenario():
        host = CapacityLoan()
        allocated = []
        original_create = loop.create_task
        def replacement(coroutine, **options):
            allocated.append(coroutine)
            return original_create(coroutine, **options)
        async def body():
            assert ownership.capture_owned_task_lender() is not None
            assert asyncio.get_running_loop() is loop
        try:
            if substitute:
                monkeypatch.setattr(loop, "create_task", replacement)
            with ownership.bind_owned_task_lender(host):
                if substitute:
                    with pytest.raises(RuntimeError, match="canonical native constructor"):
                        ownership.create_owned_task(body())
                else:
                    await ownership.create_owned_task(body())
            assert not allocated and not host.outstanding
            assert len(host.retired) == (0 if substitute else 1)
        finally:
            monkeypatch.undo()
    try:
        loop.run_until_complete(scenario())
    finally:
        loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["direct", "forwarder"])
@pytest.mark.parametrize("failed_close", [False, True])
async def test_known_source_close_failure_retains_loan_without_changing_original_error(tmp_path, profile, failed_close):
    from kestrel_sovereign._async_ownership import OwnedSourceAdmission, create_owned_iterator
    host = CapacityLoan()
    source_admission = OwnedSourceAdmission()
    original = OSError("original physical close" if failed_close else "original body refusal")
    handle = (tmp_path / "original-source-resource").open("wb")
    closes = []
    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not failed_close:
                raise original
            raise StopAsyncIteration
        async def aclose(self):
            closes.append(asyncio.current_task())
            if failed_close:
                raise original
            handle.close()
    stream = None
    try:
        with ownership.bind_owned_task_lender(host):
            if profile == "direct":
                stream = await create_owned_iterator(lambda: Source(), operation="source-retirement-proof",
                    source_admission=source_admission)
            else:
                stream = execution_stream(None, Source())
                source_admission = stream._source_admission
            with pytest.raises(BaseException) as caught:
                await anext(stream)
            assert original in execution_evidence_carriers(caught.value)
            if profile == "direct":
                assert caught.value is original
            outcome = ownership.reconcile_owned_task_admission(source_admission.producer)
            assert outcome.state == ("unknown" if failed_close else "complete")
            if failed_close:
                assert original in source_admission._state.failures
                assert original in execution_evidence_carriers(outcome.error)
                assert "_kestrel_native_retirement_debts" not in vars(original)
            assert len(host.outstanding) == (1 if failed_close else 0)
            assert len(host.retired) == (0 if failed_close else 1)
            assert handle.closed is not failed_close
            assert len(closes) == 1
    finally:
        if stream is not None:
            try:
                await stream.aclose()
            except BaseException as closed:
                assert original in execution_evidence_carriers(closed)
        handle.close()


@pytest.mark.asyncio
async def test_source_lend_veto_retires_original_partial_without_decorating_veto():
    from kestrel_sovereign._async_ownership import OwnedSourceAdmission, create_owned_iterator
    host = CapacityLoan()
    original = OSError("original source loan veto")
    host.denial = original
    source_admission = OwnedSourceAdmission()
    entered = []
    async def body():
        entered.append(asyncio.current_task())
        yield
    with ownership.bind_owned_task_lender(host):
        with pytest.raises(BaseException) as caught:
            await create_owned_iterator(body, operation="source-lend-veto", source_admission=source_admission)
        assert original in execution_evidence_carriers(caught.value)
    assert "_kestrel_native_retirement_debts" not in vars(original)
    assert "_kestrel_failed_owned_iterator" not in vars(original)
    assert not entered and not host.outstanding and len(host.retired) == 1
    assert ownership.reconcile_owned_task_admission(source_admission.producer).state == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["http-response", "http-inline", "database-facade", "database-backend"])
@pytest.mark.parametrize("fenced", [False, True])
async def test_public_physical_closers_do_not_reserve_new_work(tmp_path, resource, fenced):
    from contextlib import nullcontext
    host = CapacityLoan(capacity=0)
    scope = (bind_execution_custody(ProcessRuntimeExecutionFence("public-physical-close"))
             if fenced else nullcontext())
    if resource.startswith("http-"):
        from kestrel_sovereign.endpoints.closing_streaming_response import ClosingStreamingResponse
        handle = (tmp_path / "response-body").open("w+b")
        handle.write(b"original body")
        handle.seek(0)
        close_captures = []

        class Body:
            def __aiter__(self):
                return self

            async def __anext__(self):
                chunk = handle.read(64)
                if chunk:
                    return chunk
                raise StopAsyncIteration

            async def aclose(self):
                close_captures.append(ownership.capture_owned_task_lender())
                if fenced:
                    with pytest.raises(ExecutionAuthorityError, match="cleanup"):
                        require_execution_work()
                else:
                    require_execution_work()
                handle.close()

        body = Body()
        response = ClosingStreamingResponse(body, close_in_stream_task=resource == "http-inline")

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        try:
            with scope, ownership.bind_owned_task_lender(host):
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
                assert ownership.capture_owned_task_lender() is not None
                require_execution_work()
            assert handle.closed and close_captures and all(capture is None for capture in close_captures)
        finally:
            handle.close()
    else:
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        database = await AsyncDatabase.sqlite(str(tmp_path / "original-database.sqlite"))
        connection = database._backend._connection
        worker = getattr(connection, "_thread", connection)
        try:
            assert worker.is_alive()
            with scope, ownership.bind_owned_task_lender(host):
                await (database.close() if resource == "database-facade" else database._backend.close())
                assert ownership.capture_owned_task_lender() is not None
                require_execution_work()
            assert not worker.is_alive()
        finally:
            # Original resource disposal after a negative assertion is not an
            # acknowledgement to the host policy under test.
            await database.close()
    assert not host.outstanding and not host.retired


@pytest.mark.asyncio
async def test_join_cannot_replace_failed_original_completion_capture(tmp_path):
    receipt = ownership.OwnedTaskAdmission()
    refusal = OSError("original completion capture registration failed")
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    path = tmp_path / "unadmitted-payload"

    class UncapturedTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == "close_unstarted":
                raise refusal
            return super().add_done_callback(callback, context=context)

    def factory(loop, coroutine, **options):
        return UncapturedTask(coroutine, loop=loop, **options)

    async def body():
        path.write_text("unadmitted body ran")

    payload = body()
    try:
        loop.set_task_factory(factory)
        with pytest.raises(BaseException) as caught:
            ownership.create_owned_task(payload, admission=receipt, lender_capture=None)
    finally:
        loop.set_task_factory(previous)
    assert refusal in execution_evidence_carriers(caught.value)
    assert receipt.task is not None
    outcome = await ownership.await_owned_task(receipt.task)
    assert refusal in execution_evidence_carriers(outcome.error)
    assert receipt.task.done() and payload.cr_frame is None and not path.exists()
    assert not receipt._completion_capture_registered
    assert ownership.reconcile_owned_task_admission(receipt).state == "unknown"


@pytest.mark.asyncio
async def test_unknown_original_watcher_keeps_the_producer_loan_after_producer_join():
    from kestrel_sovereign._async_ownership import OwnedSourceAdmission, create_owned_iterator
    from kestrel_sovereign.execution_custody import execution_cleanup_debts
    host = CapacityLoan()
    source_admission = OwnedSourceAdmission()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    refusal = OSError("original watcher allocation is uncertain")
    refused = []
    entered = []
    def refuse_watcher(loop, coroutine, **options):
        refused.append(coroutine)
        raise refusal
    lend = host.lend
    def publish_then_change_factory(*args, **options):
        lend(*args, **options)
        # The producer already has its canonical native constructor. This
        # subsequent structural watcher is deliberately not a WORK loan.
        loop.set_task_factory(refuse_watcher)
    host.lend = publish_then_change_factory
    async def source():
        entered.append(asyncio.current_task())
        yield
    try:
        with ownership.bind_owned_task_lender(host):
            with pytest.raises(BaseException) as caught:
                await create_owned_iterator(source, operation="unknown-original-watcher",
                    source_admission=source_admission)
        loop.set_task_factory(previous)
        assert refusal in execution_evidence_carriers(caught.value)
        assert not entered and source_admission.producer.task.done()
        assert len(refused) == 1 and refused[0].cr_frame is None
        assert source_admission.watcher.task is None
        assert ownership.reconcile_owned_task_admission(source_admission.producer).state == "unknown"
        assert len(host.outstanding) == 1 and not host.retired
        debts = execution_cleanup_debts(caught.value)
        assert debts and all(task is not source_admission.producer.task for _, task in debts)
        assert "_kestrel_native_retirement_debts" not in vars(refusal)
    finally:
        loop.set_task_factory(previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["configured_factory", "substituted_constructor", "changed_during_reserve"])
async def test_loan_requires_canonical_constructor_before_any_borrower_allocation(boundary, monkeypatch):
    host = CapacityLoan()
    receipt = ownership.OwnedTaskAdmission()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    native_create = asyncio.create_task
    allocated = []
    def factory(loop, coroutine, **options):
        task = asyncio.Task(coroutine, loop=loop, **options)
        allocated.append(task)
        return task
    def substitute(coroutine, **options):
        task = native_create(coroutine, **options)
        allocated.append(task)
        return task
    reserve = host.reserve
    def change_during_reserve(**options):
        reservation = reserve(**options)
        loop.set_task_factory(factory)
        return reservation
    async def body():
        pytest.fail("noncanonical constructor entered a borrowed payload")
    payload = body()
    try:
        if boundary == "configured_factory":
            loop.set_task_factory(factory)
        elif boundary == "substituted_constructor":
            monkeypatch.setattr(asyncio, "create_task", substitute)
        else:
            host.reserve = change_during_reserve
        with ownership.bind_owned_task_lender(host):
            with pytest.raises(RuntimeError, match="canonical native constructor"):
                ownership.create_owned_task(payload, admission=receipt)
    finally:
        loop.set_task_factory(previous)
        monkeypatch.undo()
        if allocated:
            await asyncio.gather(*allocated, return_exceptions=True)
    assert not allocated and receipt.task is None and payload.cr_frame is None
    assert ownership.reconcile_owned_task_admission(receipt).state == "nonacceptance"
    assert not host.outstanding
    assert len(host.retired) == (1 if boundary == "changed_during_reserve" else 0), (
        "an unsupported constructor must be rejected before reserving the host resource")


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
    original_tasks = asyncio.all_tasks()
    receipt = ownership.OwnedTaskAdmission()

    async def body():
        pytest.fail("refused payload entered")

    payload = body()
    with ownership.bind_owned_task_lender(host):
        with pytest.raises(RuntimeError, match="test host capacity exhausted"):
            ownership.create_owned_task(payload, admission=receipt)
    assert asyncio.all_tasks() == original_tasks and receipt.task is None
    assert payload.cr_frame is None and not host.outstanding


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
async def test_opaque_factory_is_refused_before_borrower_acceptance():
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
            with pytest.raises(RuntimeError, match="canonical native constructor"):
                ownership.create_owned_task(body(), admission=receipt)
        finally:
            loop.set_task_factory(previous)
        assert receipt.task is None
    assert not invoked and not host.retired
    assert not host.outstanding
    assert ownership.reconcile_owned_task_admission(receipt).state == "nonacceptance"


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
            with pytest.raises(RuntimeError, match="canonical native constructor"):
                ownership.create_owned_task(body(), admission=receipt)
        finally:
            loop.set_task_factory(previous)
            await asyncio.gather(*accepted, return_exceptions=True)
    assert not entered and not accepted
    assert ownership.reconcile_owned_task_admission(receipt).state == "nonacceptance"
    assert not host.retired and not host.outstanding


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
            with pytest.raises(RuntimeError, match="canonical native constructor"):
                ownership.create_owned_task(body(), admission=receipt)
        finally:
            loop.set_task_factory(previous)
        await asyncio.gather(*accepted, return_exceptions=True)
    assert not entered and not accepted
    assert ownership.reconcile_owned_task_admission(receipt).state == "nonacceptance"
    assert not host.retired and not host.outstanding


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
                with pytest.raises(RuntimeError, match="canonical native constructor"):
                    ownership.create_owned_task(body(), admission=receipt)
            finally:
                loop.set_task_factory(previous)
        assert ownership.reconcile_owned_task_admission(receipt).state == "nonacceptance"
        assert all(task is None for _, task, _ in host.outstanding.values())

    try:
        asyncio.run(scenario())
    finally:
        if accepted:
            async def join_originals():
                await asyncio.gather(*accepted, return_exceptions=True)
            foreign.run_until_complete(join_originals())
        foreign.close()
    assert not entered and not accepted


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
async def test_detached_body_context_cannot_restore_an_old_explicit_capture():
    host = CapacityLoan()
    with ownership.bind_owned_task_lender(host):
        capture = ownership.capture_owned_task_lender()
        with without_owned_coordinators():
            detached = contextvars.copy_context()
        async def independent():
            assert ownership.capture_owned_task_lender() is None
            assert not host.outstanding
        async def body():
            await ownership.create_owned_task(independent(), lender_capture=capture)
        await ownership.create_owned_task(body(), context=detached)
    assert not host.retired


@pytest.mark.asyncio
async def test_cleanup_suppression_covers_completion_publication_in_a_fresh_context():
    host = CapacityLoan()
    children = []
    async def body():
        assert ownership.capture_owned_task_lender() is None
    with ownership.bind_owned_task_lender(host):
        capture = ownership.capture_owned_task_lender()
        def completion(_task, _error):
            children.append(ownership.create_owned_task(body(), lender_capture=capture))
        with bind_execution_custody(ProcessRuntimeExecutionFence("cleanup-publication")), bind_execution_cleanup(None):
            await ownership.create_owned_task(body(), context=contextvars.Context(), on_completion=completion)
        await asyncio.gather(*children)
    assert not host.outstanding and not host.retired


@pytest.mark.asyncio
async def test_retained_callback_rechecks_its_producers_shared_withdrawal():
    host = CapacityLoan(capacity=2)
    finalizing, release = asyncio.Event(), asyncio.Event()
    scopes, entered = [], []
    attempt = None
    async def source():
        scopes.append(OwnedCallbackScope())
        try:
            yield "original"
        finally:
            finalizing.set()
            await release.wait()
    async def work():
        entered.append(asyncio.current_task())
    with bind_execution_custody(ProcessRuntimeExecutionFence("retained-source-callback")):
        with ownership.bind_owned_task_lender(host):
            stream = execution_stream(None, source())
            assert await anext(stream) == "original"
            close = asyncio.create_task(stream.aclose())
            try:
                await asyncio.wait_for(finalizing.wait(), 1)
                with pytest.raises(ExecutionAuthorityError, match="cleanup"):
                    attempt = scopes[0].start(work, context=contextvars.Context())
            finally:
                release.set()
                await close
                if attempt is not None:
                    await asyncio.gather(attempt, return_exceptions=True)
                scopes[0].close()
    assert not entered and not host.outstanding and len(host.retired) == 1


@pytest.mark.asyncio
async def test_independent_root_inside_detachment_cannot_revive_the_old_root():
    old, new = CapacityLoan(), CapacityLoan()
    async def independent():
        assert ownership.capture_owned_task_lender() is None
        assert not old.outstanding and not new.outstanding
    async def new_work():
        assert ownership.capture_owned_task_lender() is not None
        assert not old.outstanding and len(new.outstanding) == 1
    with ownership.bind_owned_task_lender(old):
        captured = ownership.capture_owned_task_lender()
        with without_owned_coordinators():
            with ownership.bind_owned_task_lender(new):
                await ownership.create_owned_task(independent(), lender_capture=captured)
                await ownership.create_owned_task(new_work())
    assert not old.retired and len(new.retired) == 1


@pytest.mark.asyncio
async def test_already_entered_body_context_remains_a_valid_non_eager_handoff():
    context = contextvars.copy_context()
    receipt = ownership.OwnedTaskAdmission()
    marker = contextvars.ContextVar("same-entered-context", default=None)
    context.run(marker.set, "original")
    async def body():
        assert marker.get() == "original"
        return asyncio.current_task()
    payload = body()
    try:
        task = context.run(ownership.create_owned_task, payload, context=context, admission=receipt)
        assert await task is receipt.task
        assert ownership.reconcile_owned_task_admission(receipt).state == "complete"
    finally:
        if receipt.task is None:
            payload.close()


@pytest.mark.asyncio
async def test_unfenced_physical_close_needs_no_spare_work_loan(tmp_path):
    host = CapacityLoan(capacity=0)
    resource = (tmp_path / "original-unfenced-close").open("wb")
    closed = []
    class Source:
        async def aclose(self):
            require_execution_work()  # Standalone authority semantics stay unchanged.
            assert ownership.capture_owned_task_lender() is None
            resource.close()
            closed.append(asyncio.current_task())
    try:
        with ownership.bind_owned_task_lender(host):
            await close_execution_stream(None, Source())
        assert resource.closed and len(closed) == 1 and not host.retired
    finally:
        resource.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["reserve", "publication", "lend", "before_step"])
async def test_retained_work_check_stays_live_through_admission_and_first_step(phase, monkeypatch):
    from kestrel_sovereign.execution_custody import _StreamCleanupState, _bind_stream_cleanup
    host = CapacityLoan()
    state = _StreamCleanupState((ProcessRuntimeExecutionFence("lender-admission-withdrawal"),))
    lends = []
    reserve, lend = host.reserve, host.lend
    def reserve_and_withdraw(**options):
        reservation = reserve(**options)
        if phase == "reserve":
            state.requested = True
        return reservation
    def lend_and_withdraw(reservation, **options):
        lends.append(options["task"])
        lend(reservation, **options)
        if phase == "lend":
            state.requested = True
    host.reserve, host.lend = reserve_and_withdraw, lend_and_withdraw
    receipt = ownership.OwnedTaskAdmission()
    entered = []
    from contextlib import ExitStack
    from kestrel_sovereign._async_ownership import bind_owned_coordinator
    coordinator = ReentrantTransitionLock()
    original_admit = coordinator.admit_owned_task
    def withdraw_after_publication(*args):
        admitted = original_admit(*args)
        state.requested = True
        return admitted
    async def body():
        entered.append(asyncio.current_task())
    with _bind_stream_cleanup((state,)), ownership.bind_owned_task_lender(host), ExitStack() as stack:
        try:
            if phase == "publication":
                monkeypatch.setattr(coordinator, "admit_owned_task", withdraw_after_publication)
                stack.enter_context(bind_owned_coordinator(coordinator))
            with pytest.raises(ExecutionAuthorityError, match="cleanup"):
                task = ownership.create_owned_task(body(), admission=receipt)
                if phase == "before_step":
                    state.requested = True
                    await task
        finally:
            if receipt.task is not None:
                await asyncio.gather(receipt.task, return_exceptions=True)
    assert not entered and not host.outstanding and len(host.retired) == 1
    if phase == "reserve":
        assert receipt.task is None and host.retired[0].state == "nonacceptance"
    assert len(lends) == (1 if phase in {"lend", "before_step"} else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", ["direct", "privacy", "callback"])
async def test_stream_paths_use_original_producer_receipt_and_captured_root(factory):
    from kestrel_sovereign._async_ownership import OwnedSourceAdmission, create_owned_iterator

    host = CapacityLoan(capacity=1)
    receipt = OwnedSourceAdmission()
    lock = ReentrantTransitionLock()
    scope = OwnedCallbackScope()
    def start_callback(body, context, *, admission, lender_capture):
        return scope.start(body, context=context, admission=admission, lender_capture=lender_capture)
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
            owner_factory=(lock.capture_native_work_factory() if factory == "privacy" else
                           start_callback if factory == "callback" else None),
        )
        try:
            assert iterator.producer_task is receipt.producer.task
            assert await anext(iterator) == "same original reader"
        finally:
            await iterator.aclose()
    assert host.retired[0].admission is receipt.producer and not host.outstanding


@pytest.mark.asyncio
async def test_eager_factory_is_refused_without_constructing_a_borrower():
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
            with pytest.raises(RuntimeError, match="canonical native constructor"):
                ownership.create_owned_task(body(), admission=receipt)
            assert all(task is None for _, task, _ in host.outstanding.values())
        finally:
            loop.set_task_factory(previous)
            if receipt.task is not None:
                await asyncio.gather(receipt.task, return_exceptions=True)
    assert not host.outstanding and not host.retired and receipt.task is None
    assert ownership.reconcile_owned_task_admission(receipt).state == "nonacceptance"


@pytest.mark.asyncio
async def test_stream_capacity_is_checked_before_any_structural_allocation(monkeypatch):
    host = CapacityLoan(capacity=0)
    import kestrel_sovereign._async_ownership as canonical
    original_create = canonical.create_owned_task
    events = []
    original_reserve = host.reserve
    def reserve(**options):
        events.append("reserve")
        return original_reserve(**options)
    host.reserve = reserve
    def observe_structural_entry(operation, **options):
        if options.get("lender_capture") is None:
            events.append("structural-construction")
        return original_create(operation, **options)
    monkeypatch.setattr(canonical, "create_owned_task", observe_structural_entry)
    async def source():
        pytest.fail("capacity-refused producer dispatched")
        yield

    with ownership.bind_owned_task_lender(host):
        stream = execution_stream(None, source())
        try:
            with pytest.raises(RuntimeError, match="test host capacity exhausted"):
                await anext(stream)
        finally:
            await stream.aclose()
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
