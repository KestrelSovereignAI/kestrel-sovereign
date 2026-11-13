"""Core delegates source mechanics without losing either original iterator."""

from __future__ import annotations

import asyncio
import contextvars
import errno
import inspect
import os
import sys
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path

import pytest

from kestrel_sdk.native_a2a.utils.source_ownership import CallbackSource, SourceAcquisition
from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign._async_ownership import (
    OwnedAsyncIterator, OwnedCallbackSource, await_owned_task, cancel_owned_task,
    create_owned_task,
)
from kestrel_sovereign.execution_custody import (
    execution_cleanup_debts, execution_evidence_carriers, execution_native_source_error,
    execution_stream, is_execution_control_error,
)
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock


def test_callback_source_is_the_shared_sdk_contract() -> None:
    assert OwnedCallbackSource is CallbackSource


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("phase", ["owner-allocation", "owner-admission", "owner-publication",
                                  "watcher-allocation", "watcher-admission", "watcher-publication"])
async def test_resource_aware_constructor_refusal_joins_actual_accepted_tasks_before_dispatch(
    tmp_path: Path, phase: str, eager: bool,
) -> None:
    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("native eager Task requires Python 3.12")
    refusal = OSError("actual resource-aware constructor refused: " + phase)
    accepted, refused, descriptors = [], [], []
    path = tmp_path / "must-not-acquire"
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()

    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            callback_name = getattr(callback, "__name__", "")
            if ((phase == self.role + "-admission" and callback_name == "close_unstarted")
                    or (phase == "owner-publication" and self.role == "owner"
                        and callback_name == "_publish_owner_completion")
                    or (phase == "watcher-publication" and self.role == "watcher"
                        and callback_name == "_interrupt_if_watch_never_started")):
                raise refusal
            return super().add_done_callback(callback, context=context)

    def native_factory(loop, coroutine, **options):
        operation = coroutine.cr_frame.f_locals.get("operation") if coroutine.cr_frame else None
        name = getattr(getattr(operation, "cr_code", None), "co_name", "")
        role = {"run_owned": "owner", "_watch_owner_cancellation": "watcher"}.get(name)
        if role is not None:
            if phase == role + "-allocation":
                refused.append(coroutine)
                raise refusal
            if eager:
                options["eager_start"] = True
            task = PublicationTask(coroutine, loop=loop, **options)
            task.role = role
            accepted.append(task)
            return task
        return asyncio.Task(coroutine, loop=loop, **options)

    def acquire(resources: SourceAcquisition) -> AsyncIterable[bytes]:
        descriptors.append(os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600))
        pytest.fail("A refused resource-aware constructor crossed source dispatch")

    try:
        loop.set_task_factory(native_factory)
        with pytest.raises(BaseException) as caught:
            await ownership.create_owned_iterator(source_factory=acquire, operation="resource-aware admission")
        assert any(error is refusal for error in execution_evidence_carriers(caught.value))
        assert not descriptors and not path.exists()
        assert all(task.done() and inspect.getcoroutinestate(task.get_coro()) == inspect.CORO_CLOSED
                   for task in accepted)
        assert all(inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED for coroutine in refused)
        debts = execution_cleanup_debts(caught.value)
        if phase.endswith("-allocation"):
            # Our adversarial factory's private ledger is NOT an admission
            # receipt available to production. Opaque allocation refusal must
            # conservatively retain unknown acceptance, even with zero matches.
            assert any(type(resource) is ownership.OwnedTaskAdmission
                       and resource.task is task is None for resource, task in debts)
        elif phase.endswith("-admission"):
            # A failed original completion-capture registration is not repaired
            # merely by this test seeing Task.done()/a closed Python coroutine.
            assert any(type(resource) is OwnedAsyncIterator and resource.producer_task is task
                       and task in accepted for resource, task in debts)
        else:
            assert not debts, "Post-admission original constructor owners were not reconciled"
    finally:
        loop.set_task_factory(previous)
        # Negative-run recovery never substitutes for the assertions above.
        for task in accepted:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        for descriptor in descriptors:
            os.close(descriptor)


@pytest.mark.asyncio
@pytest.mark.parametrize("natural_finally", [False, True])
async def test_native_source_projection_keeps_original_failed_finalizer_and_actual_debt(
    tmp_path: Path, natural_finally: bool,
) -> None:
    path = tmp_path / "original-unretired-finalizer"
    path.write_bytes(b"still owned")
    descriptors, owners, failures = [], [], []
    delivered = guarded = source_guarded = terminal = scoped = None
    value = None
    coordinator = ReentrantTransitionLock()

    def fail_original_close() -> None:
        try:
            os.write(descriptors[0], b"real write to a read-only original FD")
        except OSError as error:
            failures.append(error)
            raise

    async def generate() -> AsyncIterator[bytes]:
        try:
            yield b"first"
        finally:
            fail_original_close()

    class Source:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            fail_original_close()

    def acquire(resources: SourceAcquisition) -> AsyncIterable[bytes]:
        descriptors.append(os.open(path, os.O_RDONLY))
        owners.append(asyncio.current_task())
        return generate() if natural_finally else Source()

    try:
        async with coordinator:
            scope = coordinator.current_native_scope()
            root = asyncio.current_task()
            try:
                async with ownership.native_source_lifetime(acquire) as reader:
                    if natural_finally:
                        value = await anext(reader)
                    try:
                        await anext(reader)
                    except BaseException as error:
                        delivered = error
                    try:
                        reader.raise_if_retirement_failed()
                    except BaseException as error:
                        guarded = error
                    try:
                        reader.raise_if_source_failed()
                    except BaseException as error:
                        source_guarded = error
            except BaseException as error:
                terminal = error
            scoped = coordinator.native_scope_control(scope, root=root)
        assert not natural_finally or value == b"first"
        assert len(failures) == 1 and failures[0].errno == errno.EBADF
        actual, = owners
        assert actual.done() and inspect.getcoroutinestate(actual.get_coro()) == inspect.CORO_CLOSED
        for label, error in (
            ("delivered", delivered), ("pre-response guard", guarded), ("request root", terminal),
            ("original source guard", source_guarded),
            ("native projection", execution_native_source_error(terminal)), ("native scope", scoped),
        ):
            assert error is not None, label
            assert is_execution_control_error(error), label
            assert any(item is failures[0] for item in execution_evidence_carriers(error)), label
            debt, = execution_cleanup_debts(error)
            resource, producer = debt
            assert type(resource) is OwnedAsyncIterator and resource.producer_task is producer is actual
        assert failures[0].__cause__ is None
        assert os.fstat(descriptors[0]).st_size == len(b"still owned")
    finally:
        # Recover deliberately failed test cleanup, not an original-close ACK.
        for descriptor in descriptors:
            os.close(descriptor)


@pytest.mark.asyncio
async def test_native_request_join_retains_every_delivered_withdrawal_without_rewriting_originals(
    tmp_path: Path,
) -> None:
    closing, release = asyncio.Event(), asyncio.Event()
    delivered_events = [asyncio.Event(), asyncio.Event()]
    delivered, originals, retired, descriptors = [], [], [], []
    request_task = None
    join_code = ownership.await_owned_task.__wrapped__.__code__
    previous_trace = sys.gettrace()
    marker = contextvars.ContextVar("native-withdrawal-finalizer", default="outside")

    async def generate():
        descriptor = os.open(tmp_path / "withdrawal-source", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        descriptors.append(descriptor)
        originals.append(asyncio.current_task())
        token = marker.set("original")
        try:
            yield b"first"
        finally:
            try:
                closing.set()
                await release.wait()
                assert marker.get() == "original"
                os.close(descriptor)
                retired.append(asyncio.current_task())
            finally:
                marker.reset(token)

    async def request():
        coordinator = ReentrantTransitionLock()
        try:
            async with coordinator:
                async with ownership.native_source_lifetime(lambda resources: generate()) as reader:
                    assert await anext(reader) == b"first"
        except BaseException as error:
            return error

    def trace(frame, event, arguments):
        if frame.f_code is not join_code:
            return None
        if event == "exception" and asyncio.current_task() is request_task:
            error = arguments[1]
            if type(error) is asyncio.CancelledError and error.args in [("withdrawal-one",), ("withdrawal-two",)]:
                delivered.append(error)
                delivered_events[len(delivered) - 1].set()
        return trace

    try:
        sys.settrace(trace)
        request_task = asyncio.create_task(request())
        await closing.wait()
        request_task.cancel("withdrawal-one")
        await delivered_events[0].wait()
        request_task.cancel("withdrawal-two")
        await delivered_events[1].wait()
        release.set()
        terminal = await request_task
        evidence = execution_evidence_carriers(terminal)
        assert len(delivered) == 2 and delivered[0] is not delivered[1]
        assert all(any(error is item for item in evidence) for error in delivered)
        assert all(error.__cause__ is None for error in delivered)
        assert originals == retired and len(originals) == 1
        actual, = originals
        assert actual.done() and actual.cancelling() == 0
        assert inspect.getcoroutinestate(actual.get_coro()) == inspect.CORO_CLOSED
        assert marker.get() == "outside" and execution_native_source_error(terminal) is None
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
    finally:
        release.set()
        sys.settrace(previous_trace)
        if request_task is not None:
            await request_task


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_finally", [False, True])
async def test_finite_host_observation_retains_actual_native_request_source_until_physical_join(
    tmp_path: Path, fail_finally: bool,
) -> None:
    closing, release = asyncio.Event(), asyncio.Event()
    descriptors, original_tasks, retired_tasks, failures, sources, admissions = [], [], [], [], [], []
    marker = contextvars.ContextVar("finite-native-source", default="outside")

    async def generate():
        descriptor = os.open(tmp_path / "finite-source", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        descriptors.append(descriptor)
        original_tasks.append(asyncio.current_task())
        token = marker.set("original")
        try:
            yield b"first"
        finally:
            try:
                closing.set()
                await release.wait()
                assert marker.get() == "original"
                os.close(descriptor)
                retired_tasks.append(asyncio.current_task())
                if fail_finally:
                    try:
                        os.fstat(descriptor)
                    except OSError as error:
                        failures.append(error)
                        raise
            finally:
                marker.reset(token)

    async def request():
        coordinator = ReentrantTransitionLock()
        async with coordinator:
            async with ownership.native_source_lifetime(lambda resources: generate()) as reader:
                assert await anext(reader) == b"first"
                sources.append(reader._source)
                admissions.extend(coordinator.current_native_scope().admissions)

    request_task = asyncio.create_task(request())
    try:
        async with asyncio.timeout(5):
            await closing.wait()
        source, = sources
        physical = (source.producer_task, source.owner_task)
        assert original_tasks == [source.producer_task]
        assert all(task in admissions for task in physical)
        observed = await ownership.wait_owned_tasks_until(
            (request_task, *physical), deadline=asyncio.get_running_loop().time() + .01,
        )
        assert observed.result == (set(), {request_task, *physical})
        assert observed.error is observed.cancellation is None
        debt = ownership.pending_owned_retirement_error(
            source, physical, message="original native request source remains physically pending",
        )
        assert execution_cleanup_debts(debt) == tuple((source, task) for task in physical)
        assert retired_tasks == [] and marker.get() == "outside"
        assert all(not task.done() and task.cancelling() == 0 for task in (request_task, *physical))
        os.fstat(descriptors[0])
        release.set()
        outcome = await await_owned_task(request_task)
        assert original_tasks == retired_tasks
        assert all(task.done() and inspect.getcoroutinestate(task.get_coro()) == inspect.CORO_CLOSED
                   for task in (request_task, *physical))
        assert execution_cleanup_debts(debt) == tuple((source, task) for task in physical)
        if fail_finally:
            assert len(failures) == 1 and failures[0].errno == errno.EBADF
            assert any(error is failures[0] for error in execution_evidence_carriers(outcome.error))
            assert any(error is failures[0]
                       for error in execution_evidence_carriers(execution_native_source_error(outcome.error)))
            assert is_execution_control_error(outcome.error)
        else:
            assert outcome.error is outcome.cancellation is None
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
    finally:
        release.set()
        await await_owned_task(request_task)


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["neither", "both", "bad-legacy", "bad-source"])
async def test_source_factory_choice_is_validated_before_native_task_admission(choice: str) -> None:
    original_tasks = asyncio.all_tasks()

    def forbidden() -> AsyncIterable[bytes]:
        pytest.fail("Invalid factory selection must not dispatch any source work")

    options = {
        "neither": {},
        "both": {"iterator_factory": forbidden, "source_factory": lambda resources: forbidden()},
        "bad-legacy": {"iterator_factory": object()},
        "bad-source": {"source_factory": object()},
    }[choice]
    with pytest.raises(TypeError):
        OwnedAsyncIterator(operation="refused source factory", **options)
    assert asyncio.all_tasks() == original_tasks


@pytest.mark.asyncio
async def test_unread_native_root_does_not_dispatch_a_withdrawn_factory() -> None:
    acquired: list[SourceAcquisition] = []

    def acquire(resources: SourceAcquisition) -> AsyncIterable[bytes]:
        acquired.append(resources)
        pytest.fail("A known stopped root must not acquire a source")

    coordinator = ReentrantTransitionLock()
    async with coordinator:
        scope = coordinator.current_native_scope()
        assert scope is not None
        async with ownership.native_source_lifetime(acquire) as reader:
            assert not hasattr(reader, "aclose")
            reader.raise_if_source_failed()
            originals = tuple(scope.admissions)
            assert len(originals) == 2
        for task in originals:
            assert task.done()
            assert inspect.getcoroutinestate(task.get_coro()) == inspect.CORO_CLOSED
    assert acquired == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["factory", "aiter"])
@pytest.mark.parametrize("fail_close", [False, True])
@pytest.mark.parametrize("request_root", [False, True])
async def test_resource_aware_factory_retains_partial_acquisition_in_original_owner(
    tmp_path: Path, failure_phase: str, fail_close: bool, request_root: bool,
) -> None:
    original = ValueError("original partial acquisition")
    cause = LookupError("original acquisition cause")
    marker = contextvars.ContextVar("partial native acquisition", default="outside")
    descriptors: list[int] = []
    phases: list[tuple[str, asyncio.Task | None]] = []
    close_errors: list[OSError] = []
    records: list[SourceAcquisition] = []

    def acquire(resources: SourceAcquisition) -> AsyncIterable[bytes]:
        records.append(resources)
        phases.append(("factory", asyncio.current_task()))
        token = marker.set("original")
        descriptor = os.open(tmp_path / "partial", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        descriptors.append(descriptor)

        class Source:
            def __aiter__(self) -> AsyncIterator[bytes]:
                assert marker.get() == "original"
                phases.append(("aiter", asyncio.current_task()))
                raise original from cause

            async def aclose(self) -> None:
                assert marker.get() == "original"
                phases.append(("close", asyncio.current_task()))
                os.close(descriptor)
                marker.reset(token)
                if fail_close:
                    try:
                        os.fstat(descriptor)
                    except OSError as error:
                        close_errors.append(error)
                        raise

        resource = Source()
        resources.retain(resource, resource.aclose)
        if failure_phase == "factory":
            raise original from cause
        return resource

    async def consume(reader: AsyncIterator[bytes]) -> BaseException:
        with pytest.raises(BaseException) as demand:
            await anext(reader)
        return demand.value

    source = None
    try:
        if request_root:
            lifetime = ownership.native_source_lifetime
            with pytest.raises(BaseException) as root:
                async with lifetime(acquire) as reader:
                    assert not hasattr(reader, "aclose")
                    delivered = await consume(reader)
            observed = root.value
        else:
            source = OwnedAsyncIterator(source_factory=acquire, operation="partial source factory")
            delivered = await consume(source)
            with pytest.raises(BaseException) as close:
                await source.aclose()
            observed = close.value
        evidence = execution_evidence_carriers(observed)
        assert any(error is original for error in evidence)
        assert any(error is original for error in execution_evidence_carriers(delivered))
        assert original.__cause__ is cause
        assert len(records) == 1
        assert [phase for phase, _ in phases] == (
            ["factory", "close"] if failure_phase == "factory" else ["factory", "aiter", "close"]
        )
        actual = phases[0][1]
        assert actual is not None
        assert all(task is actual for _, task in phases)
        assert actual.done()
        assert inspect.getcoroutinestate(actual.get_coro()) == inspect.CORO_CLOSED
        if source is not None:
            assert actual is source.producer_task and source.owner_task.done()
        assert marker.get() == "outside"
        if fail_close:
            assert len(close_errors) == 1 and close_errors[0].errno == errno.EBADF
            assert any(error is close_errors[0] for error in evidence)
        else:
            assert close_errors == []
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
    finally:
        if source is not None and not source.producer_task.done():
            await source.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_close", [False, True])
@pytest.mark.parametrize("forwarded", [False, True])
async def test_resource_aware_opaque_finally_is_not_aborted_by_close(
    tmp_path: Path, fail_close: bool, forwarded: bool,
) -> None:
    closing, release = asyncio.Event(), asyncio.Event()
    marker = contextvars.ContextVar("opaque original finally", default="outside")
    descriptors: list[int] = []
    owners: list[asyncio.Task | None] = []
    retired: list[asyncio.Task | None] = []
    errors: list[OSError] = []

    async def generate() -> AsyncIterator[str]:
        descriptor = os.open(tmp_path / "opaque-finally", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        descriptors.append(descriptor)
        owners.append(asyncio.current_task())
        token = marker.set("original")
        try:
            yield "first"
        finally:
            try:
                closing.set()
                await release.wait()
                assert marker.get() == "original"
                os.close(descriptor)
                retired.append(asyncio.current_task())
                if fail_close:
                    try:
                        os.fstat(descriptor)
                    except OSError as error:
                        errors.append(error)
                        raise
            finally:
                marker.reset(token)

    source = OwnedAsyncIterator(
        source_factory=lambda resources: execution_stream(None, generate()) if forwarded else generate(),
        operation="opaque original finally",
    )
    reader = closer = None
    try:
        assert await anext(source) == "first"
        reader = create_owned_task(anext(source), retain_originals=True)
        await closing.wait()
        closer = create_owned_task(source.aclose(), retain_originals=True)
        completed, _ = await asyncio.wait((closer,), timeout=.01)
        assert not completed
        assert not source.producer_task.done()
        assert source.producer_task.cancelling() == 0
        assert retired == []
        os.fstat(descriptors[0])
        release.set()
        close_outcome, read_outcome = await await_owned_task(closer), await await_owned_task(reader)
        if fail_close:
            assert len(errors) == 1 and errors[0].errno == errno.EBADF
            assert close_outcome.error is read_outcome.error is source.terminal_error
            assert any(error is errors[0] for error in execution_evidence_carriers(close_outcome.error))
            assert is_execution_control_error(close_outcome.error)
        else:
            assert close_outcome.error is close_outcome.cancellation is None
            assert isinstance(read_outcome.error, StopAsyncIteration)
        assert len(owners) == 1 and owners == retired
        original = owners[0]
        assert original is not None and original.done()
        assert (original is source.producer_task) is not forwarded
        assert inspect.getcoroutinestate(original.get_coro()) == inspect.CORO_CLOSED
        assert source.producer_task.done() and source.owner_task.done()
        assert inspect.getcoroutinestate(source.producer_task.get_coro()) == inspect.CORO_CLOSED
        assert marker.get() == "outside"
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
    finally:
        release.set()
        if reader is not None:
            await await_owned_task(reader)
        if closer is not None:
            await await_owned_task(closer)
        else:
            await source.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("request_root", [False, True])
async def test_failed_resource_close_retains_actual_owner_debt_not_task_done_ack(
    tmp_path: Path, request_root: bool,
) -> None:
    path = tmp_path / "unretired-source"
    path.write_bytes(b"real source")
    descriptors: list[int] = []
    owners: list[asyncio.Task | None] = []
    failures: list[OSError] = []

    def acquire(resources: SourceAcquisition) -> AsyncIterable[bytes]:
        descriptor = os.open(path, os.O_RDONLY)
        descriptors.append(descriptor)
        owners.append(asyncio.current_task())

        class Source:
            def __aiter__(self) -> Source:
                return self

            async def __anext__(self) -> bytes:
                raise StopAsyncIteration

            async def aclose(self) -> None:
                try:
                    os.write(descriptor, b"cannot write to the real read-only FD")
                except OSError as error:
                    failures.append(error)
                    raise

        return Source()

    source = None
    try:
        if request_root:
            with pytest.raises(BaseException) as root:
                async with ownership.native_source_lifetime(acquire) as reader:
                    with pytest.raises(BaseException):
                        await anext(reader)
            terminal = root.value
        else:
            source = OwnedAsyncIterator(source_factory=acquire, operation="real unretired source")
            with pytest.raises(BaseException) as delivered:
                await anext(source)
            terminal = delivered.value
        assert len(failures) == 1 and failures[0].errno == errno.EBADF
        assert any(error is failures[0] for error in execution_evidence_carriers(terminal))
        assert is_execution_control_error(terminal), "A failed original close is native retirement control"
        debts = execution_cleanup_debts(terminal)
        assert len(debts) == 1
        resource, actual = debts[0]
        assert type(resource) is OwnedAsyncIterator
        assert resource.producer_task is actual is owners[0]
        assert actual.done()
        assert inspect.getcoroutinestate(actual.get_coro()) == inspect.CORO_CLOSED
        assert failures[0].__cause__ is None
        # A physically finished Task is NOT a successful resource-close ACK.
        assert os.fstat(descriptors[0]).st_size == len(b"real source")
    finally:
        # Explicit test recovery of the deliberately failed closer. This is
        # neither its original retirement receipt nor a second aclose attempt.
        for descriptor in descriptors:
            os.close(descriptor)
        if source is not None and not source.producer_task.done():
            await source.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_early", [False, True])
async def test_distinct_outer_and_inner_retire_in_the_original_source(
    tmp_path: Path, close_early: bool,
) -> None:
    path = tmp_path / "original-source"
    path.write_bytes(b"x")
    current = contextvars.ContextVar("shared-source-context", default="outside")
    descriptors: list[int] = []
    original: list[asyncio.Task] = []
    retired: list[tuple[str, asyncio.Task | None, str]] = []

    def acquire() -> AsyncIterable[bytes]:
        owner = asyncio.current_task()
        assert owner is not None
        original.append(owner)
        token = current.set("original")
        outer_fd = os.open(path, os.O_RDONLY)
        try:
            inner_fd = os.open(path, os.O_RDONLY)
        except OSError:
            os.close(outer_fd)
            current.reset(token)
            raise
        descriptors.extend((outer_fd, inner_fd))

        class Inner:
            def __aiter__(self) -> Inner:
                return self

            async def __anext__(self) -> bytes:
                value = os.read(inner_fd, 1)
                if not value:
                    raise StopAsyncIteration
                return value

            async def aclose(self) -> None:
                retired.append(("inner", asyncio.current_task(), current.get()))
                os.close(inner_fd)
                current.reset(token)

        inner = Inner()

        class Outer:
            def __aiter__(self) -> Inner:
                return inner

            async def aclose(self) -> None:
                retired.append(("outer", asyncio.current_task(), current.get()))
                os.close(outer_fd)

        return Outer()

    source = OwnedAsyncIterator(acquire, operation="shared-original-source")
    try:
        assert await anext(source) == b"x"
        if not close_early:
            with pytest.raises(StopAsyncIteration):
                await anext(source)
        await source.aclose()
        assert original == [source.producer_task]
        assert source.producer_task.done()
        for descriptor in descriptors:
            with pytest.raises(OSError) as captured:
                os.fstat(descriptor)
            assert captured.value.errno == errno.EBADF
        assert retired == [
            ("inner", source.producer_task, "original"),
            ("outer", source.producer_task, "outside"),
        ]
        assert current.get() == "outside"
    finally:
        await source.aclose()
        # Test-only cleanup keeps the negative pre-fix run from leaking an FD.
        # It is not an original-owner retirement receipt or a second aclose.
        for descriptor in descriptors:
            try:
                os.fstat(descriptor)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
            else:
                os.close(descriptor)


@pytest.mark.asyncio
async def test_source_and_failure_classifier_errors_remain_original(
    tmp_path: Path,
) -> None:
    path = tmp_path / "original-resource"
    path.write_bytes(b"x")
    body_error = ValueError("original source failure")
    body_graph = execution_evidence_carriers(body_error)
    predicate_errors: list[OSError] = []
    predicate_graphs: list[tuple[BaseException, ...]] = []
    descriptors: list[int] = []
    retired: list[asyncio.Task | None] = []

    def cleanup_requested() -> bool:
        try:
            with (tmp_path / "missing-predicate-state").open("rb"):
                raise AssertionError("The predicate state must be absent")
        except FileNotFoundError as error:
            predicate_errors.append(error)
            predicate_graphs.append(execution_evidence_carriers(error))
            raise

    class Source:
        def __aiter__(self) -> Source:
            return self

        async def __anext__(self) -> bytes:
            raise body_error

        async def aclose(self) -> None:
            os.close(descriptors[0])
            retired.append(asyncio.current_task())

    def acquire() -> Source:
        descriptors.append(os.open(path, os.O_RDONLY))
        return Source()

    source = OwnedAsyncIterator(
        acquire, operation="original-source-and-classifier-errors",
        cleanup_requested=cleanup_requested,
    )
    try:
        with pytest.raises(BaseException) as captured:
            await anext(source)
        evidence = execution_evidence_carriers(captured.value)
        assert any(error is body_error for error in evidence)
        assert len(predicate_errors) == 1
        assert any(error is predicate_errors[0] for error in evidence)
        assert execution_evidence_carriers(body_error) == body_graph
        assert execution_evidence_carriers(predicate_errors[0]) == predicate_graphs[0]
        assert source.producer_task.done()
        assert retired == [source.producer_task]
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_private_interrupt_is_rechecked_after_original_retirement() -> None:
    entered = asyncio.Event()
    descriptors: list[int] = []
    interrupts: list[asyncio.CancelledError] = []
    physical_errors: list[OSError] = []

    class Source:
        def __aiter__(self) -> Source:
            return self

        async def __anext__(self) -> bytes:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                interrupts.append(error)
                raise
            raise AssertionError("The original source must be interrupted")

        async def aclose(self) -> None:
            os.close(descriptors[0])
            try:
                os.fstat(descriptors[0])
            except OSError as error:
                assert error.errno == errno.EBADF
                physical_errors.append(error)
                # Some closers enrich the original body failure rather than
                # throw a second error. It is no longer a causeless private C.
                interrupts[0].__cause__ = error

    def acquire() -> Source:
        descriptors.append(os.open(os.devnull, os.O_RDONLY))
        return Source()

    source = OwnedAsyncIterator(
        acquire, operation="private-interrupt-with-late-original-cause",
        cleanup_requested=lambda: True,
    )
    try:
        await entered.wait()
        assert cancel_owned_task(source.producer_task)
        outcome = await await_owned_task(source.producer_task)
        assert len(physical_errors) == 1
        assert any(error is physical_errors[0]
                   for error in execution_evidence_carriers(outcome.error))
        with pytest.raises(BaseException) as observed:
            await source.aclose()
        assert any(error is physical_errors[0]
                   for error in execution_evidence_carriers(observed.value))
        assert source.producer_task.done() and source.owner_task.done()
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_original_closer_error_graphs_are_not_rewritten() -> None:
    descriptors: list[int] = []
    failures: list[tuple[OSError, tuple[BaseException, ...]]] = []
    retired: list[asyncio.Task | None] = []

    def close_original(index: int) -> None:
        os.close(descriptors[index])
        retired.append(asyncio.current_task())
        try:
            os.fstat(descriptors[index])
        except OSError as error:
            assert error.errno == errno.EBADF
            failures.append((error, execution_evidence_carriers(error)))
            raise
        raise AssertionError("The original descriptor remained open")

    class Inner:
        def __aiter__(self) -> Inner:
            return self

        async def __anext__(self) -> bytes:
            return b"original frame"

        async def aclose(self) -> None:
            close_original(1)

    class Outer:
        def __aiter__(self) -> Inner:
            return Inner()

        async def aclose(self) -> None:
            close_original(0)

    def acquire() -> Outer:
        first = os.open(os.devnull, os.O_RDONLY)
        try:
            second = os.open(os.devnull, os.O_RDONLY)
        except OSError:
            os.close(first)
            raise
        descriptors.extend((first, second))
        return Outer()

    source = OwnedAsyncIterator(acquire, operation="immutable-original-close-errors")
    try:
        assert await anext(source) == b"original frame"
        with pytest.raises(BaseException) as captured:
            await source.aclose()
        evidence = execution_evidence_carriers(captured.value)
        assert len(failures) == 2
        for error, original_graph in failures:
            assert any(carrier is error for carrier in evidence)
            assert execution_evidence_carriers(error) == original_graph
        assert retired == [source.producer_task, source.producer_task]
        assert source.producer_task.done() and source.owner_task.done()
    finally:
        await source.aclose()
        for descriptor in descriptors:
            try:
                os.fstat(descriptor)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
            else:
                os.close(descriptor)


@pytest.mark.asyncio
async def test_concurrent_close_cannot_acknowledge_a_pending_original_finally(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pending-original-close"
    path.write_bytes(b"x")
    closing, release = asyncio.Event(), asyncio.Event()
    descriptors: list[int] = []
    retired: list[asyncio.Task | None] = []
    closers: list[asyncio.Task] = []

    class Source:
        def __aiter__(self) -> Source:
            return self

        async def __anext__(self) -> bytes:
            value = os.read(descriptors[0], 1)
            if not value:
                raise StopAsyncIteration
            return value

        async def aclose(self) -> None:
            closing.set()
            await release.wait()
            os.close(descriptors[0])
            retired.append(asyncio.current_task())

    def acquire() -> Source:
        descriptors.append(os.open(path, os.O_RDONLY))
        return Source()

    source = OwnedAsyncIterator(acquire, operation="concurrent-original-close")
    try:
        assert await anext(source) == b"x"
        closers.append(asyncio.create_task(source.aclose()))
        await closing.wait()
        closers.append(asyncio.create_task(source.aclose()))
        completed, _ = await asyncio.wait(closers, timeout=.01)
        assert not completed, "A closer ACKed while the original finally still owns its FD"
        assert not source.producer_task.done()
        os.fstat(descriptors[0])
        release.set()
        assert await asyncio.gather(*closers) == [None, None]
        assert retired == [source.producer_task]
        assert source.producer_task.done() and source.owner_task.done()
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
    finally:
        release.set()
        if closers:
            await asyncio.gather(*closers)
        await source.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("withdraw_first", [False, True])
async def test_overlapping_close_retains_failure_after_both_original_owners_finish(
    withdraw_first: bool,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    descriptors: list[int] = []
    physical_errors: list[OSError] = []
    physical_closers: list[asyncio.Task | None] = []
    closers: list[asyncio.Task] = []
    second_entry = {}

    class Source:
        def __aiter__(self) -> Source:
            return self

        async def __anext__(self) -> bytes:
            return b"original frame"

        async def aclose(self) -> None:
            physical_closers.append(asyncio.current_task())
            entered.set()
            await release.wait()
            os.close(descriptors[0])
            try:
                os.fstat(descriptors[0])
            except OSError as error:
                assert error.errno == errno.EBADF
                physical_errors.append(error)
                raise

    def acquire() -> Source:
        descriptors.append(os.open(os.devnull, os.O_RDONLY))
        return Source()

    source = OwnedAsyncIterator(acquire, operation="overlapping-close-original-failure")

    async def observed_close():
        try:
            await source.aclose()
        except BaseException as error:
            return error
        return None

    try:
        assert await anext(source) == b"original frame"
        first = asyncio.create_task(observed_close())
        closers.append(first)
        await entered.wait()
        if withdraw_first:
            first.cancel("original-first-close-withdrawal")

        async def close_after_physical_completion():
            while not (source.producer_task.done() and source.owner_task.done()):
                await asyncio.sleep(0)
            second_entry["first_pending"] = not first.done()
            return await observed_close()

        closers.append(asyncio.create_task(close_after_physical_completion()))
        release.set()
        first_error, second_error = await asyncio.gather(*closers)
        assert second_entry == {"first_pending": True}
        assert source.producer_task.done() and source.owner_task.done()
        assert physical_closers == [source.producer_task]
        assert len(physical_errors) == 1
        original, = physical_errors
        assert original in execution_evidence_carriers(first_error)
        assert original in execution_evidence_carriers(second_error), (
            "An overlapping close ACKed the same failed original retirement"
        )
        if withdraw_first:
            assert any(isinstance(error, asyncio.CancelledError)
                and error.args == ("original-first-close-withdrawal",)
                for error in execution_evidence_carriers(first_error))
        assert not any(isinstance(error, asyncio.CancelledError)
            and error.args == ("original-first-close-withdrawal",)
            for error in execution_evidence_carriers(second_error))
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
        # A sequential repeat after all close calls returned remains quiet;
        # it may not rerun the physical closer or re-raise an observed failure.
        await source.aclose()
        assert physical_closers == [source.producer_task]
    finally:
        release.set()
        if closers:
            await asyncio.gather(*closers, return_exceptions=True)
        await source.aclose()
        for descriptor in descriptors:
            try:
                os.fstat(descriptor)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
            else:
                os.close(descriptor)
