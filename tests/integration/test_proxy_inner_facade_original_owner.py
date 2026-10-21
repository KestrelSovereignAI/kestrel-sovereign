"""Actual inner lifecycle admission and serialized retirement keep originals."""
import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.features import isolated_runtime
from kestrel_sovereign.execution_custody import _explicit_control_chain
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.unit.test_isolated_feature_runtime import (
    FakeIsolatedClient, _initialized_host_ingress_proxy,
)
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["refuse", "publish-before", "publish-after"])
async def test_inner_facade_admission_keeps_actual_native_opening(
    native_pg, monkeypatch, fault,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    accepted, refused, fired, late = [], [], [], []
    async def stop():
        raise AssertionError("publication-failed admission executed foreign stop")
    operation = stop()
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *args, **kwargs):
            if not fired and getattr(callback, "__name__", "") == "close_unstarted":
                fired.append(True)
                if fault == "publish-after":
                    super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        actual = _actual_operation(coroutine)
        if getattr(actual, "cr_code", None) is stop.__code__:
            if fault == "refuse":
                fired.append(True)
                refused.extend((coroutine, actual))
                raise source.error
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    async def run():
        await isolated_runtime._await_owned_facade_lifecycle_operation(
            operation, name="original-inner-facade", on_late_task=late.append,
        )
    owner = create_owned_task(run(), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        outcome = await await_owned_task(owner)
        assert fired, "inner facade never used original canonical admission seam"
        source.assert_retained(outcome.error)
        assert all(task.done() for task in accepted), "accepted inner owner was abandoned"
        assert inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED, "inner facade leaked refused actual coroutine"
        assert not late
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for coroutine in refused:
            coroutine.close()
        operation.close()


@pytest.mark.asyncio
async def test_inner_facade_native_failure_keeps_actual_finite_withdrawal(
    native_pg, monkeypatch,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    entered, release, withdrawals, admitted = asyncio.Event(), asyncio.Event(), [], []
    native_wait = asyncio.wait_for
    async def observed_wait(*args, **kwargs):
        try:
            return await native_wait(*args, **kwargs)
        except asyncio.CancelledError as error:
            withdrawals.append(error)
            raise
    async def stop():
        entered.set()
        await release.wait()
        raise source.error
    async def run():
        await isolated_runtime._await_owned_facade_lifecycle_operation(
            stop(), name="finite-original-inner-stop", on_admitted=admitted.append,
            on_late_task=lambda operation: pytest.fail("finite original was detached"),
        )
    monkeypatch.setattr(asyncio, "wait_for", observed_wait)
    owner = create_owned_task(run(), retain_originals=True)
    try:
        await native_wait(entered.wait(), 5)
        owner.cancel("actual inner lifecycle observer withdrew")
        await asyncio.sleep(0)
        assert not owner.done()
        release.set()
        outcome = await await_owned_task(owner)
        source.assert_retained(outcome.error)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert withdrawals and withdrawals[0] in chain, "inner lifecycle dropped actual finite caller cancellation"
        source.assert_retained(admitted[0]._host_task._kestrel_owned_terminal_error)
        assert withdrawals[0] not in tuple(_explicit_control_chain(admitted[0]._host_task._kestrel_owned_terminal_error))
    finally:
        release.set()
        await await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_queued_terminal_retirement_never_retries_failed_native_stop(
    native_pg, tmp_path, monkeypatch, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    entered, release, stops = asyncio.Event(), asyncio.Event(), []
    class Client(FakeIsolatedClient):
        async def stop(self):
            stops.append(asyncio.current_task())
            entered.set()
            await release.wait()
            raise source.error
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, Client)
    client = feature._client
    feature._retain_terminal_retirement_client(client)
    owners = []
    try:
        owners.append(create_owned_task(feature._retire_terminal_clients(), retain_originals=True))
        await asyncio.wait_for(entered.wait(), 5)
        owners.append(create_owned_task(feature._retire_terminal_clients(retry_retirement=True), retain_originals=True))
        await asyncio.sleep(0)
        assert not owners[1].done(), "fixture did not queue a second physical retirement owner"
        release.set()
        outcomes = [await await_owned_task(owner) for owner in owners]
        assert len(stops) == 1, "queued owner retried an unresolved original native stop"
        markers = [marker for marker in feature._terminal_stop_completed_client_markers if marker.client() is client]
        assert len(markers) == 1 and markers[0].operation is not None
        assert markers[0].operation._host_task is stops[0] and stops[0].done()
        source.assert_retained(markers[0].failure)
        source.assert_retained(feature._event_ingress_control)
        assert outcomes[1].result is False
    finally:
        release.set()
        for owner in owners:
            await await_owned_task(owner)
        supervisor = feature._supervision_task
        if supervisor is not None:
            feature._supervision_task = None
            supervisor.cancel()
            await await_owned_task(supervisor)
