"""Finite proxy observers do not become shared lifecycle withdrawal history."""
import asyncio

import pytest

from kestrel_sovereign.features import isolated_runtime
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from tests.integration.test_completion_observer_consumers import _publication_task_type
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.unit.test_isolated_feature_runtime import FakeIsolatedClient, _initialized_host_ingress_proxy
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_proxy_native_observation_keeps_finite_actual_withdrawal(
    native_pg, tmp_path, monkeypatch, owned, remove, after, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    entered, release, accepted, withdrawals, stops = asyncio.Event(), asyncio.Event(), [], [], []
    class Client(FakeIsolatedClient):
        async def stop(self):
            stops.append(asyncio.current_task())
            await native_pg[0].execute("INSERT INTO effects VALUES (245, 'actual proxy cleanup hook')")
            entered.set()
            await release.wait()
            await super().stop()
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, Client)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    task_type = _publication_task_type(source.error, remove=remove, after=after)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is isolated_runtime.ProxyFeature._terminal_cleanup_uninterrupted.__code__:
            task = task_type(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    native_join = isolated_runtime.await_owned_task
    async def join(task, *args, **kwargs):
        result = await native_join(task, *args, **kwargs)
        if task in accepted and result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    monkeypatch.setattr(isolated_runtime, "await_owned_task", join)
    async def run():
        if owned:
            feature._latch_terminal_lifecycle()
            async with feature._reload_lock:
                await feature._complete_terminal_cleanup(lifecycle_lock_held=True)
        else:
            await feature.shutdown()
    loop.set_task_factory(factory)
    caller = create_owned_task(run(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        caller.cancel("actual original proxy observer withdrew")
        await asyncio.sleep(0)
        assert not caller.done(), "proxy observer abandoned original physical cleanup"
        release.set()
        outcome = await await_owned_task(caller)
        source.assert_retained(outcome.error)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert withdrawals and all(error in chain for error in withdrawals), "proxy discarded actual finite observer withdrawal"
        assert len(accepted) == len(stops) == 1 and accepted[0].done()
        assert await native_pg[1].fetchval(f'SELECT count(*) FROM "{native_pg[2]}".effects WHERE id=245') == 1
        repeated = await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
        source.assert_retained(repeated.error)
        assert all(error not in tuple(_explicit_control_chain(repeated.error)) for error in withdrawals), "later proxy observer inherited caller cancellation"
        assert len(accepted) == len(stops) == 1
    finally:
        loop.set_task_factory(previous)
        release.set()
        await await_owned_task(caller)
        for task in accepted:
            await await_owned_task(task)
