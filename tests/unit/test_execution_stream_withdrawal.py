"""Explicit cooperative forwarding keeps the actual original Task/context alive."""

import asyncio
import errno
import os
from contextvars import ContextVar
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from kestrel_sdk.native_a2a.utils.source_ownership import SourceWithdrawal

from kestrel_sovereign.execution_custody import execution_evidence_carriers, owned_execution_stream
from kestrel_sovereign import _async_ownership as ownership


@pytest.fixture
def original_withdrawals(monkeypatch):
    """Observe actual delivered errors without replacing their native owner."""
    state = SimpleNamespace(task=None, errors=[], arrived=[asyncio.Event(), asyncio.Event()])
    original = asyncio.wait

    async def observe(*args, **kwargs):
        try:
            return await original(*args, **kwargs)
        except asyncio.CancelledError as error:
            if asyncio.current_task() is state.task and error.args in [("withdraw-one",), ("withdraw-two",)]:
                state.errors.append(error)
                state.arrived[len(state.errors) - 1].set()
            raise

    monkeypatch.setattr(asyncio, "wait", observe)
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["legacy-close", "close", "signal"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("eager", [False, True])
async def test_reserved_source_retires_without_a_consumer_pull(trigger, fail, eager):
    import sqlite3

    from kestrel_sovereign.execution_custody import is_execution_control_error

    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("native eager Task requires Python 3.12")
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    signal = None if trigger == "legacy-close" else SourceWithdrawal()
    native = sqlite3.connect(":memory:")
    descriptors = os.pipe()
    advanced, closed, physical = [], [], []
    closer = None
    assertions_finished = False
    recovering = False

    class AcquiredOriginal:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if recovering:
                raise StopAsyncIteration
            advanced.append(asyncio.current_task())
            pytest.fail("Reserved source advanced without an admitted consumer pull")

        async def aclose(self):
            closed.append(asyncio.current_task())
            native.close()
            for descriptor in descriptors:
                os.close(descriptor)
            if fail:
                try:
                    os.fstat(descriptors[0])
                except OSError as error:
                    physical.append(error)
                    raise

    try:
        if eager:
            loop.set_task_factory(asyncio.eager_task_factory)
        async with owned_execution_stream(object(), AcquiredOriginal(), withdrawal=signal) as events:
            async def close():
                try:
                    await events.aclose()
                except BaseException as error:
                    return error

            try:
                await events.reserve()
                producer = events._iterator.producer_task
                for _ in range(20):
                    await asyncio.sleep(0)
                assert events._producer_started and not advanced and not closed
                if trigger == "signal":
                    signal.request_stop()
                    target = producer
                else:
                    closer = asyncio.create_task(close())
                    target = closer
                done, _ = await asyncio.wait((target,), timeout=0.2)
                if not done:
                    # Capture the real leaked resource BEFORE test-only recovery.
                    assert native.execute("SELECT 1").fetchone() == (1,)
                    assert not closed
                assert done, "Reserved consumer gate ignored cooperative withdrawal"
                if closer is None:
                    closer = asyncio.create_task(close())
                terminal = await closer
                if fail:
                    assert len(physical) == 1
                    if signal is not None:
                        assert is_execution_control_error(terminal)
                    assert any(node is physical[0] for node in execution_evidence_carriers(terminal))
                else:
                    assert terminal is None
                assert closed == [producer] and producer.done() and not advanced
                if signal is not None:
                    assert producer.cancelling() == 0
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    native.in_transaction
                for descriptor in descriptors:
                    with pytest.raises(OSError) as error:
                        os.fstat(descriptor)
                    assert error.value.errno == errno.EBADF
                assertions_finished = True
            finally:
                if not assertions_finished:
                    # Recover only the known resource-free TEST gate after the
                    # actual stuck-owner/open-C evidence, never a product ACK.
                    recovering = True
                    events._consumer_started.set()
                if closer is not None:
                    await asyncio.gather(closer, return_exceptions=True)
    except BaseException as error:
        if not assertions_finished:
            raise
        assert fail and any(node is physical[0] for node in execution_evidence_carriers(error))
    finally:
        loop.set_task_factory(previous_factory)
        native.close()
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError as error:
                assert error.errno == errno.EBADF


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["request", "forwarder", "iterator", "callback"])
async def test_cancelled_original_finally_is_not_a_private_stop_ack(route):
    from kestrel_sovereign.execution_custody import execution_cleanup_debts, is_execution_control_error

    closing, release = asyncio.Event(), asyncio.Event()
    signal = SourceWithdrawal()
    descriptors, owners, cancelled, retired, readers = [], [], [], [], []
    task = None

    async def original():
        owners.append(asyncio.current_task())
        descriptors.extend(os.pipe())
        try:
            yield "first"
        finally:
            closing.set()
            try:
                await release.wait()
            except asyncio.CancelledError as error:
                cancelled.append(error)
                raise
            for descriptor in descriptors:
                os.close(descriptor)
            retired.append(asyncio.current_task())

    async def callback(emit):
        async for value in original():
            await emit(value)

    @asynccontextmanager
    async def iterator_lifetime():
        source = ownership.OwnedAsyncIterator(
            source_factory=lambda resources: ownership.OwnedCallbackSource(callback) if route == "callback" else original(),
            operation="actual-original-finally-cancellation",
        )
        try:
            yield source
        finally:
            await source.aclose()

    async def request():
        lifetime = (
            ownership.native_source_lifetime(lambda resources: original()) if route == "request"
            else owned_execution_stream(object(), original(), withdrawal=signal) if route == "forwarder"
            else iterator_lifetime()
        )
        try:
            async with lifetime as reader:
                readers.append(reader)
                assert await anext(reader) == "first"
                try:
                    await anext(reader)
                except StopAsyncIteration:
                    pass
        except BaseException as error:
            return error

    try:
        async with asyncio.timeout(5):
            task = asyncio.create_task(request())
            await closing.wait()
            signal.request_stop() if route == "forwarder" else readers[0].request_stop()
            assert ownership.cancel_owned_task(owners[0])
            terminal = await task
            assert isinstance(terminal, BaseException), "Failed original finally was acknowledged as successful retirement"
            assert is_execution_control_error(terminal) and execution_cleanup_debts(terminal)
            assert len(cancelled) == 1 and not retired
            assert any(error is cancelled[0] for error in execution_evidence_carriers(terminal))
            assert len(owners) == 1 and owners[0].done() and owners[0].cancelling() == 1
            # A later no-op close cannot certify the failed original finally.
            for descriptor in descriptors:
                os.fstat(descriptor)
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        # Recover exact test descriptors AFTER failed-retirement evidence. This
        # is not an original physical close ACK and cannot satisfy assertions.
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError as error:
                assert error.errno == errno.EBADF


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["observer", "retirement", "unstarted"])
@pytest.mark.parametrize("fail", [False, True])
async def test_stream_close_retains_every_actual_withdrawal(original_withdrawals, phase, fail):
    state = original_withdrawals
    signal = SourceWithdrawal()
    closing, release = asyncio.Event(), asyncio.Event()
    descriptors, owners, retired, physical = [], [], [], []
    closer = None
    assertions_finished = False

    async def retire_original():
        closing.set()
        await release.wait()
        for descriptor in descriptors:
            os.close(descriptor)
        retired.append(asyncio.current_task())
        if fail:
            try:
                os.fstat(descriptors[0])
            except OSError as error:
                physical.append(error)
                raise

    async def original():
        owners.append(asyncio.current_task())
        descriptors.extend(os.pipe())
        try:
            yield "first"
            await signal.wait()
        finally:
            await retire_original()

    class AcquiredOriginal:
        def __init__(self):
            descriptors.extend(os.pipe())

        def __aiter__(self):
            return self

        async def __anext__(self):
            pytest.fail("Unstarted original source was advanced during close")

        async def aclose(self):
            owners.append(asyncio.current_task())
            await retire_original()

    try:
        source = AcquiredOriginal() if phase == "unstarted" else original()
        async with owned_execution_stream(object(), source, withdrawal=signal) as events:
            async def close():
                try:
                    await events.aclose()
                except BaseException as error:
                    return error

            try:
                async with asyncio.timeout(5):
                    if phase != "unstarted":
                        assert await anext(events) == "first"
                    closer = asyncio.create_task(close())
                    await closing.wait()
                    state.task = closer if phase == "observer" else events.retirement_task
                    state.task.cancel("withdraw-one")
                    await state.arrived[0].wait()
                    state.task.cancel("withdraw-two")
                    await state.arrived[1].wait()
                    assert not closer.done() and owners[0].cancelling() == 0
                    release.set()
                    terminal = await closer
                    graph = tuple(execution_evidence_carriers(terminal))
                    assert len(state.errors) == 2 and state.errors[0] is not state.errors[1]
                    assert all(any(error is node for node in graph) for error in state.errors)
                    assert all(error.__cause__ is None for error in state.errors)
                    assert owners == retired and owners[0].done() and owners[0].cancelling() == 0
                    if fail:
                        assert len(physical) == 1 and any(node is physical[0] for node in graph)
                    for descriptor in descriptors:
                        with pytest.raises(OSError) as closed:
                            os.fstat(descriptor)
                        assert closed.value.errno == errno.EBADF
                    fresh = None
                    try:
                        await events.aclose()
                    except BaseException as error:
                        fresh = error
                    fresh_graph = tuple(execution_evidence_carriers(fresh))
                    assert all(any(error is node for node in fresh_graph) == (phase != "observer") for error in state.errors)
                    assertions_finished = True
            finally:
                release.set()
    except BaseException as error:
        if not assertions_finished or phase == "observer" and not fail:
            raise
        graph = tuple(execution_evidence_carriers(error))
        if phase != "observer":
            assert all(any(original is node for node in graph) for original in state.errors)
        else:
            assert any(node is physical[0] for node in graph)
    finally:
        release.set()
        if closer is not None:
            await asyncio.gather(closer, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("retain", [False, True])
async def test_failed_owner_publication_preserves_explicit_withdrawal_policy(original_withdrawals, retain):
    state = original_withdrawals
    entered, release = asyncio.Event(), asyncio.Event()
    descriptors, owners, retired, physical = [], [], [], []
    observer = None

    async def original():
        owners.append(asyncio.current_task())
        descriptors.extend(os.pipe())
        entered.set()
        try:
            await release.wait()
        finally:
            for descriptor in descriptors:
                os.close(descriptor)
            retired.append(asyncio.current_task())

    def publish(operation, context, admission, resource):
        ownership.create_owned_task(operation, context=context, admission=admission, retirement_resource=resource, retain_originals=True)
        descriptor = os.open(os.devnull, os.O_WRONLY)
        try:
            try:
                os.read(descriptor, 1)
            except OSError as error:
                physical.append(error)
                raise
        finally:
            os.close(descriptor)

    operation = ownership.OwnedOperation(original(), retirement_resource=object(), retain_originals=True, owner_factory=publish)
    try:
        async with asyncio.timeout(5):
            observer = asyncio.create_task(operation.observe(retain_cancellations=retain))
            state.task = observer
            await entered.wait()
            observer.cancel("withdraw-one")
            await state.arrived[0].wait()
            observer.cancel("withdraw-two")
            await state.arrived[1].wait()
            assert not observer.done() and operation.task.cancelling() == 0
            release.set()
            outcome = await observer
            assert owners == retired == [operation.task] and operation.task.done()
            assert any(node is physical[0] for node in execution_evidence_carriers(outcome.error))
            graph = tuple(execution_evidence_carriers(outcome.cancellation))
            assert [any(error is node for node in graph) for error in state.errors] == [True, retain]
            assert all(error.__cause__ is None for error in state.errors)
            later = await operation.observe(retain_cancellations=retain)
            assert later.cancellation is None
            assert not any(error is node for error in state.errors for node in execution_evidence_carriers(later.error))
            for descriptor in descriptors:
                with pytest.raises(OSError) as closed:
                    os.fstat(descriptor)
                assert closed.value.errno == errno.EBADF
    finally:
        release.set()
        if observer is not None:
            await asyncio.gather(observer, return_exceptions=True)
        if operation.task is not None:
            await ownership.await_owned_task(operation.task)


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["idle", "retiring"])
@pytest.mark.parametrize("close_kind", ["observer", "scope"])
@pytest.mark.parametrize("fail", [False, True])
async def test_explicit_withdrawal_preserves_actual_finalizer_task_context_and_fds(trigger, close_kind, fail):
    signal = SourceWithdrawal()
    waiting, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    value = ContextVar("original-source-token", default=None)
    descriptors, owners, retired, cancelled, physical = [], [], [], [], []
    pull, closer = None, None
    assertions_finished = False

    async def original():
        owner = asyncio.current_task()
        owners.append(owner)
        token = value.set(owner)
        descriptors.extend(os.pipe())
        try:
            yield "first"
            waiting.set()
            await signal.wait()
        finally:
            closing.set()
            try:
                await release.wait()
            except asyncio.CancelledError as error:
                cancelled.append(error)
                raise
            assert asyncio.current_task() is owner and owner.cancelling() == 0
            assert value.get() is owner
            value.reset(token)
            for fd in descriptors:
                os.close(fd)
            retired.append(owner)
            if fail:
                try:
                    os.fsync(descriptors[0])
                except OSError as error:
                    physical.append(error)
                    raise

    try:
        async with owned_execution_stream(object(), original(), withdrawal=signal) as events:
            try:
                async with asyncio.timeout(5):
                    assert await anext(events) == "first" and value.get() is None
                pull = asyncio.create_task(anext(events))
                async with asyncio.timeout(5):
                    await waiting.wait()
                if trigger == "retiring":
                    signal.request_stop()
                    async with asyncio.timeout(5):
                        await closing.wait()
                if close_kind == "observer":
                    pull.cancel()
                else:
                    closer = asyncio.create_task(events.aclose())
                async with asyncio.timeout(5):
                    await closing.wait()
                await asyncio.wait((pull,) if closer is None else (pull, closer), timeout=0.02)
                assert not retired and not cancelled
                assert len(owners) == 1 and not owners[0].done() and owners[0].cancelling() == 0
                for fd in descriptors:
                    os.fstat(fd)
                release.set()
                async with asyncio.timeout(5):
                    results = await asyncio.gather(*[task for task in (pull, closer) if task is not None], return_exceptions=True)
                assert retired == owners and not cancelled and value.get() is None
                if fail:
                    assert len(physical) == 1 and physical[0].errno == errno.EBADF
                    assert any(
                        error is physical[0]
                        for result in results if isinstance(result, BaseException)
                        for error in execution_evidence_carriers(result)
                    )
                assertions_finished = True
            finally:
                # Recovery MUST release before the lexical owner joins close.
                signal.request_stop()
                release.set()
    except BaseException as error:
        graph = execution_evidence_carriers(error)
        if not assertions_finished or not fail or not physical or not any(node is physical[0] for node in graph):
            raise
    finally:
        # Explicit failed-baseline cleanup only; it cannot satisfy assertions.
        signal.request_stop()
        release.set()
        pending = [task for task in (pull, closer) if task is not None]
        if pending:
            async with asyncio.timeout(5):
                await asyncio.gather(*pending, return_exceptions=True)
        for fd in descriptors:
            try:
                os.fstat(fd)
            except OSError as error:
                assert error.errno == errno.EBADF
            else:
                os.close(fd)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["plain", "derived"])
async def test_bad_withdrawal_refuses_before_iterator_acquisition(kind):
    acquired = []

    class DerivedWithdrawal(SourceWithdrawal):
        pass

    class Source:
        def __aiter__(self):
            acquired.append(True)
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    with pytest.raises(TypeError, match="SourceWithdrawal"):
        async with owned_execution_stream(object(), Source(), withdrawal=object() if kind == "plain" else DerivedWithdrawal()):
            pytest.fail("Invalid withdrawal entered an execution source")
    assert not acquired
