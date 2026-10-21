"""Published-ready rollback returns actual finite observer evidence."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_native_source_error
from kestrel_sovereign import _async_ownership as ownership_module
from kestrel_sovereign.multi_agent import agent_manager as manager_module
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
async def test_published_ready_rollback_keeps_actual_observer_cancellation(
    native_pg, onboarding_owner, monkeypatch,
):
    manager, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    admission, owns = await manager._admit_agent_operation("RollbackReady", kind="load")
    assert owns
    manager._register_agent("RollbackReady", agent)
    admission.published = True
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    actual_join = ownership_module.await_owned_task

    async def join(*args, **kwargs):
        result = await actual_join(*args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result

    async def ready(_agent):
        raise source.error

    async def remove(name, **_kwargs):
        entered.set()
        await release.wait()
        await database.execute("INSERT INTO host_rollback_effects VALUES (317)")
        manager._withdraw_initialized_agent(name, agent)
        return True

    monkeypatch.setattr(ownership_module, "await_owned_task", join)
    monkeypatch.setattr(manager_module, "await_owned_task", join)
    monkeypatch.setattr(manager, "_run_hosted_agent_ready_hooks", ready)
    monkeypatch.setattr(manager, "remove_agent", remove)
    owner = create_owned_task(manager._finish_published_agent_readiness(
        "RollbackReady", agent, admission, deadline=None,
        failure_description="native ready boundary and rollback",
    ), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual published rollback observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        shared_sources = tuple(_explicit_control_chain(execution_native_source_error(result.error)))
        assert all(error not in shared_sources for error in withdrawals)
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=317") == 1
        assert manager.get_agent("RollbackReady") is None and not admission.published
    finally:
        release.set()
        await await_owned_task(owner)
        manager._withdraw_initialized_agent("RollbackReady", agent)
        await manager._release_agent_operation(admission)
