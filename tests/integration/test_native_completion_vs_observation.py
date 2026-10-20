"""Physical SQLite close and finite observation failure are separate facts."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task, pending_owned_retirement_error
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
from kestrel_sovereign.host_features.context import _close_partial_host_resources
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.native_admission_fault import create_admission_task
from tests.utils.original_failure import OriginalFailure

onboarding_owner = _onboarding_owner


def observation_factory(predicate, failure, attached, accepted, previous, *, eager):
    class ObservationTask(asyncio.Task):
        failed = False

        def add_done_callback(self, callback, *args, **kwargs):
            result = super().add_done_callback(callback, *args, **kwargs)
            if getattr(callback, "__name__", "") == "mark_complete":
                attached.set()
            return result

        def remove_done_callback(self, callback):
            result = super().remove_done_callback(callback)
            if not self.failed and self.done() and getattr(callback, "__name__", "") == "mark_complete":
                self.failed = True
                raise failure
            return result

    def factory(loop, coroutine, **kwargs):
        if predicate(_actual_operation(coroutine)):
            task = create_admission_task(ObservationTask, coroutine, loop=loop, eager=eager, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    return factory


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
async def test_actual_admission_task_constructor_preserves_requested_start(eager):
    entered = []

    class OriginalTask(asyncio.Task):
        pass

    async def operation():
        entered.append(asyncio.current_task())
        return "original completion"

    task = create_admission_task(
        OriginalTask, operation(), loop=asyncio.get_running_loop(), eager=eager,
    )
    try:
        assert type(task) is OriginalTask
        assert bool(entered) is eager
        assert await task == "original completion"
        assert entered == [task]
    finally:
        await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
async def test_shutdown_completion_keeps_actual_observation_error_finite_after_native_close(
    onboarding_owner, eager,
):
    _manager, agent, db = onboarding_owner
    connection = db.backend._connection
    worker = getattr(connection, "_thread", connection)
    entered, release, attached = asyncio.Event(), asyncio.Event(), asyncio.Event()
    failure = OSError("actual original completion callback removal failed")
    accepted = []
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()

    async def close():
        await db.close()
        entered.set()
        await release.wait()

    loop.set_task_factory(observation_factory(
        lambda operation: getattr(operation, "cr_code", None) is close.__code__,
        failure, attached, accepted, previous, eager=eager,
    ))
    closer = create_owned_task(close(), retain_originals=True)
    loop.set_task_factory(previous)
    waiter = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        source = OriginalFailure.capture(pending_owned_retirement_error(connection, [closer], message="actual original close ACK pending"))
        agent._remember_shutdown_control(source.error)
        waiter = create_owned_task(agent.wait_for_shutdown_completion(), retain_originals=True)
        await asyncio.wait_for(attached.wait(), 5)
        release.set()
        result = await await_owned_task(waiter)
        source.assert_retained(result.error)
        assert failure in tuple(_explicit_control_chain(result.error))
        assert failure not in tuple(_explicit_control_chain(agent._shutdown_control))
        assert closer.done() and not worker.is_alive()
        later = await await_owned_task(create_owned_task(agent.wait_for_shutdown_completion(), retain_originals=True))
        source.assert_retained(later.error)
        assert failure not in tuple(_explicit_control_chain(later.error))
    finally:
        release.set()
        if waiter is not None:
            await await_owned_task(waiter)
        await await_owned_task(closer)


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
async def test_host_close_success_is_not_reclassified_as_failed_retirement_by_observer(
    monkeypatch, eager,
):
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    connection = backend._connection
    worker = getattr(connection, "_thread", connection)
    native_close, calls = backend.close, []
    failure = OSError("actual host close completion observer failed")
    attached, accepted, owners = asyncio.Event(), [], {}
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()

    async def close():
        calls.append(asyncio.current_task())
        await native_close()

    monkeypatch.setattr(backend, "close", close)
    factory = observation_factory(lambda operation: getattr(operation, "cr_code", None) is close.__code__, failure, attached, accepted, previous, eager=eager)
    owner = create_owned_task(_close_partial_host_resources(None, None, None, backend, owners=owners), retain_originals=True)
    try:
        loop.set_task_factory(factory)
        result = await await_owned_task(owner)
        loop.set_task_factory(previous)
        assert result.error is failure
        assert accepted and accepted[0].done() and calls == accepted
        assert owners[id(backend)].closed and owners[id(backend)].task is None and owners[id(backend)].error is None
        assert backend._connection is None and not worker.is_alive()
        assert execution_cleanup_debts(result.error) == ()
        await _close_partial_host_resources(None, None, None, backend, owners=owners)
        assert calls == accepted, "healthy later observation admitted a replacement closer"
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        await native_close()
