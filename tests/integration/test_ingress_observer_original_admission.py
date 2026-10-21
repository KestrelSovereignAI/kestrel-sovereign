"""Actual bounded queue observers retire without taking over ingress delivery.

The input is an existing real asyncio queue/worker and local PostgreSQL work.
It is a component admission contract, not a launched child or delivery ACK.
"""

import asyncio
import inspect
import secrets
from types import SimpleNamespace

import pytest

from kestrel_sovereign import execution_custody
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.features import isolated_runtime
from kestrel_sovereign.features.isolated_runtime import (
    ProxyFeature,
    _NonCursorIngressQueue,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,kind",
    [
        ("admission", "ordinary"),
        ("admission", "native-control"),
        ("admission", "native-cancel"),
        ("withdrawal", "ordinary"),
        ("withdrawal", "native-control"),
        ("withdrawal", "native-cancel"),
        ("join", "healthy"),
        ("retired", "healthy"),
        ("retired", "native-control"),
    ],
)
async def test_bounded_ingress_observers_join_without_owning_source(
    native_pg, monkeypatch, mode, kind
):
    backend, observer, schema = native_pg
    await backend.execute("CREATE TABLE ingress_worker_effect (id INTEGER PRIMARY KEY)")
    async with backend.advisory_locks([(3569, secrets.randbelow(2**30))]) as lease:
        lease.require_live()
    started, release, admitted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    unwinding, settle_observer = asyncio.Event(), asyncio.Event()
    originals, withdrawals, accepted, refused = [], [], [], []

    def original_failure():
        if kind == "ordinary":
            error = OSError("actual observer admission refused")
        else:
            try:
                lease.require_live()
            except BaseException as native:
                error = native
            if kind == "native-cancel":
                cancellation = asyncio.CancelledError("native observer cancelled")
                cancellation.__cause__ = error
                error = cancellation
        originals.append(error)
        return error

    class NativeQueue(asyncio.Queue):
        async def join(self):
            try:
                return await super().join()
            finally:
                if mode != "admission" and kind.startswith("native"):
                    unwinding.set()
                    await settle_observer.wait()
                    raise original_failure()

    queue = NativeQueue(maxsize=1)
    original_event, new_event = object(), object()
    queue.put_nowait(original_event)
    client = object()
    entry = _NonCursorIngressQueue(client=client, events=queue)

    async def native_worker():
        await backend.execute("INSERT INTO ingress_worker_effect VALUES (1)")
        started.set()
        await release.wait()

    worker = create_owned_task(native_worker())
    entry.worker = worker
    gate = SimpleNamespace(closed=False)
    feature = SimpleNamespace(
        _terminal_lifecycle_latched=False,
        _traffic_gate=gate,
        _client=client,
        _non_cursor_event_ingress_lock=asyncio.Lock(),
        _non_cursor_event_ingress_queues=[entry],
        name="actual-local-ingress-component",
    )
    await asyncio.wait_for(started.wait(), 5)
    actual_create = asyncio.create_task

    def create(operation, *args, **kwargs):
        original_operation = operation
        frame = operation.cr_frame
        if frame is not None and inspect.iscoroutine(frame.f_locals.get("operation")):
            original_operation = frame.f_locals["operation"]
        original_frame = original_operation.cr_frame
        receiver = original_frame.f_locals.get("self") if original_frame else None
        selected = receiver is queue or receiver is entry.retired
        if selected and mode == "admission" and len(accepted) == 1:
            refused.extend((operation, original_operation))
            raise original_failure()
        task = actual_create(operation, *args, **kwargs)
        if selected:
            accepted.append(task)
            if len(accepted) == 2:
                admitted.set()
        return task

    monkeypatch.setattr(asyncio, "create_task", create)

    actual_terminal = execution_custody.execution_terminal_error

    def terminal(*errors, **kwargs):
        if asyncio.current_task() is caller:
            for error in errors:
                if (
                    isinstance(error, asyncio.CancelledError)
                    and error.args == ("actual ingress observer withdrawal",)
                    and error not in withdrawals
                ):
                    withdrawals.append(error)
        return actual_terminal(*errors, **kwargs)

    monkeypatch.setattr(execution_custody, "execution_terminal_error", terminal)
    actual_gather = isolated_runtime.gather_owned
    observers_joined = asyncio.Event()

    async def gather(*args, **kwargs):
        try:
            return await actual_gather(*args, **kwargs)
        finally:
            observers_joined.set()

    monkeypatch.setattr(isolated_runtime, "gather_owned", gather)

    async def enqueue():
        try:
            await ProxyFeature._enqueue_non_cursor_inbound_event(
                feature, new_event, client
            )
        except asyncio.CancelledError as cancellation:
            if cancellation.args == ("actual ingress observer withdrawal",):
                withdrawals.append(cancellation)
            raise

    caller = create_owned_task(enqueue(), retain_originals=True)
    unwind_waiter = None
    try:
        if mode != "admission":
            await asyncio.wait_for(admitted.wait(), 5)
            await asyncio.sleep(0)
            if mode == "withdrawal":
                caller.cancel("actual ingress observer withdrawal")
                if kind == "ordinary":
                    await asyncio.wait_for(observers_joined.wait(), 5)
                    # If a mutant swallows cancellation and retries pressure,
                    # close this finite source instead of hanging the test.
                    gate.closed = True
                    entry.retired.set()
            elif mode == "join":
                assert queue.get_nowait() is original_event
                queue.task_done()
            else:
                entry.accepting = False
                gate.closed = True
                entry.retired.set()
            if kind.startswith("native"):
                unwind_waiter = create_owned_task(unwinding.wait())
                await asyncio.wait_for(
                    asyncio.wait(
                        (caller, unwind_waiter), return_when=asyncio.FIRST_COMPLETED
                    ),
                    5,
                )
                assert unwinding.is_set() and not caller.done(), (
                    "ingress returned before its original observer physically settled"
                )
                settle_observer.set()
        outcome = await asyncio.wait_for(await_owned_task(caller), 5)
        assert all(task.done() for task in accepted), (
            "bounded ingress observer abandoned its original accepted peer"
        )
        assert all(
            inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED
            for operation in refused
        ), "bounded ingress admission leaked its actual refused coroutine"
        if kind != "healthy":
            assert outcome.error is not None
            chain = tuple(_explicit_control_chain(outcome.error))
            if mode == "withdrawal":
                assert withdrawals and withdrawals[0] in chain
            if kind.startswith("native") or mode == "admission":
                assert originals and all(error in chain for error in originals)
            if kind.startswith("native"):
                assert is_execution_control_error(outcome.error)
        else:
            assert outcome.error is None
        # Observer cleanup must not cancel, replace, or ACK the source worker.
        assert entry.worker is worker and not worker.done()
        assert (
            await observer.fetchval(
                f'SELECT count(*) FROM "{schema}".ingress_worker_effect'
            )
            == 1
        )
        if mode != "join":
            assert queue.get_nowait() is original_event
        else:
            assert queue.get_nowait() is new_event
        queue.task_done()
    finally:
        # Before-fix runs clean only their exact leaked observers.
        gate.closed = True
        entry.retired.set()
        settle_observer.set()
        if unwind_waiter is not None:
            if not unwind_waiter.done():
                unwind_waiter.cancel()
            await await_owned_task(unwind_waiter)
        for operation in refused:
            if inspect.getcoroutinestate(operation) == inspect.CORO_CREATED:
                operation.close()
        for task in accepted:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        if not caller.done():
            caller.cancel()
        await await_owned_task(caller)
        release.set()
        await await_owned_task(worker)
