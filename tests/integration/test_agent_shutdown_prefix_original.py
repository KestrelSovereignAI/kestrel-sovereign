"""Agent prefix admission retains original native control before durable close."""
import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure
from tests.shared.native_shutdown_budget import reserve_native_shutdown_budget


@pytest.fixture(autouse=True)
def native_prefix_budget(onboarding_owner, monkeypatch):
    _manager, agent, _database = onboarding_owner
    # These tests must enter a real fallible prefix before the durable tail.
    # CI reserves a larger SQLite checkpoint window than local defaults. Keep
    # that complete native reservation AND five seconds for the prefix instead
    # of letting a fixed five-second total silently allocate zero prefix time.
    reserve_native_shutdown_budget(agent, monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_shutdown_prefix_refusal_joins_accepted_original_then_closes_storage(
    native_pg, onboarding_owner, monkeypatch, phase, source_cancel,
):
    _manager, agent, database = onboarding_owner
    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    source = OriginalFailure.capture(original)
    effects, accepted, refused = [], [], []
    connection = database.backend._connection
    worker = getattr(connection, "_thread", connection)
    class NativePrefix:
        async def shutdown(self):
            await database.execute("INSERT INTO host_rollback_effects VALUES (4)")
            effects.append(True)
    feature = NativePrefix()
    agent.features = {"NativePrefix": feature}
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    class PublicationTask(asyncio.Task):
        refused = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not self.refused and getattr(callback, "__name__", "") == "close_unstarted":
                self.refused = True
                if phase == "publish-after":
                    super().add_done_callback(callback, *args, **kwargs)
                raise original
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is NativePrefix.shutdown.__code__:
            if phase == "refuse":
                refused.append((coroutine, operation))
                raise original
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    owner = create_owned_task(agent.shutdown(), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        outcome = await await_owned_task(owner)
        source.assert_retained(outcome.error)
        assert len(refused if phase == "refuse" else accepted) == 1
        assert not effects, "publication-failed prefix spent native work"
        assert all(task.done() for task in accepted), "prefix discarded accepted original shutdown Task"
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair), "prefix admission leaked an unstarted original coroutine"
        assert database.backend._connection is None and not worker.is_alive()
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("already_interrupted", [False, True])
async def test_shutdown_prefix_repeated_observer_cancel_joins_native_unwind(
    native_pg, onboarding_owner, monkeypatch, source_cancel, already_interrupted,
):
    _manager, agent, database = onboarding_owner
    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    source = OriginalFailure.capture(original)
    entered, unwinding, joining, release = (asyncio.Event() for _ in range(4))
    attempts, withdrawals = [], []
    connection = database.backend._connection
    worker = getattr(connection, "_thread", connection)
    class NativePrefix:
        async def shutdown(self):
            attempts.append(asyncio.current_task())
            await database.execute("INSERT INTO host_rollback_effects VALUES (5)")
            entered.set()
            if already_interrupted:
                asyncio.current_task().cancel("original prefix was already interrupted")
            try:
                await asyncio.Event().wait()
            finally:
                unwinding.set()
                await release.wait()
                raise original
    from kestrel_sovereign import _async_ownership as ownership_module
    actual_join = ownership_module.await_owned_task
    async def join(task, *args, **kwargs):
        if getattr(_actual_operation(task.get_coro()), "cr_code", None) is NativePrefix.shutdown.__code__:
            joining.set()
        outcome = await actual_join(task, *args, **kwargs)
        if task in attempts and outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome
    monkeypatch.setattr(ownership_module, "await_owned_task", join)
    agent.features = {"NativePrefix": NativePrefix()}
    owner = create_owned_task(agent.shutdown(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("first actual prefix observer cancellation")
        await asyncio.wait_for(unwinding.wait(), 5)
        owner.cancel("second actual prefix observer cancellation")
        await asyncio.wait_for(joining.wait(), 5)
        owner.cancel("third actual original join observer cancellation")
        await asyncio.sleep(0)
        assert not owner.done(), "prefix reported completion while its original native unwind was held"
        release.set()
        outcome = await await_owned_task(owner)
        source.assert_retained(outcome.error)
        from kestrel_sovereign.execution_custody import _explicit_control_chain
        assert withdrawals and all(error in tuple(_explicit_control_chain(outcome.error)) for error in withdrawals), "prefix discarded actual original observer withdrawal"
        assert len(attempts) == 1 and attempts[0].done()
        assert database.backend._connection is None and not worker.is_alive()
    finally:
        release.set()
        await await_owned_task(owner)
        for task in attempts:
            await await_owned_task(task)
