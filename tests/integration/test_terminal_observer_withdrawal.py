"""Finite terminal observers preserve withdrawal outside shared native latches.

Actual local PG control and actual resident SQLite work are used. This is not
full Agent READY, a started cloud scheduler, provider inference or deployment.
"""
import asyncio

import pytest

import kestrel_sovereign._async_ownership as ownership_module
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.features.health import feature as health_module
from kestrel_sovereign.features.scheduler import runner as scheduler_module
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_completion_observer_consumers import _publication_task_type
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["feature", "health", "scheduler"])
@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_terminal_observer_withdrawal_is_finite_and_source_is_immutable(native_pg, onboarding_owner, monkeypatch, surface, remove, source_cancel):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    _manager, agent, database = onboarding_owner
    entered, release, joining = asyncio.Event(), asyncio.Event(), asyncio.Event()
    originals, withdrawals = [], []
    class ResidentFeature(Feature):
        @property
        def tool_description(self):
            return "Actual native finite observer"
        async def initialize(self):
            return None
    feature = ResidentFeature(agent)
    health = health_module.HealthFeature(agent)
    health._health_persist_tasks = set()  # Declared pre-READY persistence inventory.
    runner = scheduler_module.SchedulerRunner(AsyncDatabase(native_pg[0]), agent.did, lambda *args: None)
    async def work():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        await database.execute("INSERT INTO host_rollback_effects VALUES (51)")
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    task_type = _publication_task_type(source.error, remove=remove, after=True)
    def factory(loop, coroutine, **kwargs):
        if getattr(_actual_operation(coroutine), "cr_code", None) is work.__code__:
            task = task_type(coroutine, loop=loop, **kwargs)
            originals.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    actual_join, owner = ownership_module.await_owned_task, None
    async def join(task, *args, **kwargs):
        if asyncio.current_task() is owner:
            joining.set()
        outcome = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome
    monkeypatch.setattr(ownership_module, "await_owned_task", join)
    monkeypatch.setattr(health_module, "await_owned_task", join)
    monkeypatch.setattr(scheduler_module, "await_owned_task", join)
    loop.set_task_factory(factory)
    try:
        if surface == "feature":
            feature._track_owned_background_task(work(), name="finite-native-feature")
            retire = feature._cancel_owned_background_tasks
        else:
            task = create_owned_task(work(), retain_originals=True)
            if surface == "health":
                health._background_task = task
                retire = health.shutdown
            else:
                runner._worker_task = task
                retire = runner.stop
        await asyncio.wait_for(entered.wait(), 5)
        owner = create_owned_task(retire(), retain_originals=True)
        await asyncio.wait_for(joining.wait(), 5)
        owner.cancel("actual terminal observer withdrawal")
        await asyncio.sleep(0)
        assert not owner.done()
        release.set()
        observed = await await_owned_task(owner)
        source.assert_retained(observed.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(observed.error)) for error in withdrawals)
        shared = {"feature": vars(feature).get("_owned_background_control"), "health": vars(health).get("_health_persist_control"), "scheduler": vars(runner).get("_execution_control_error")}[surface]
        source.assert_retained(shared)
        assert all(error not in tuple(_explicit_control_chain(shared)) for error in withdrawals)
        assert len(originals) == 1 and originals[0].done() and originals[0].cancelling() == 1
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=51") == 1
    finally:
        loop.set_task_factory(previous)
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        for task in originals:
            await await_owned_task(task)
