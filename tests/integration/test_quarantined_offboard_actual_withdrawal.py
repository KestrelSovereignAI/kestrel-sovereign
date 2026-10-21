"""Real quarantined namespace retirement carries the actual observer C."""
import asyncio
import threading

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.features import isolated_runtime as runtime_module
from kestrel_sovereign.multi_agent import agent_manager as manager_module
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("physical_failure", [False, True])
async def test_quarantined_offboard_keeps_actual_withdrawal_after_native_delete(
    native_pg, onboarding_owner, monkeypatch, physical_failure,
):
    manager, agent, _database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    scope = runtime_module.resolve_isolated_runtime_namespace(
        manager._isolated_runtime_root,
        runtime_module.derive_isolated_runtime_namespace(agent.did),
    )
    runtime_module.prepare_isolated_runtime_namespace(scope, agent.did)
    agent.isolated_runtime_scope = scope
    entered, release, withdrawals = asyncio.Event(), threading.Event(), []
    loop = asyncio.get_running_loop()
    native_remove = runtime_module.remove_agent_runtime_namespace
    actual_join = manager_module.await_owned_task

    def remove(candidate):
        result = native_remove(candidate)
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "fixture did not release actual filesystem retirement ACK"
        if physical_failure:
            raise source.error
        return result

    async def join(*args, **kwargs):
        result = await actual_join(*args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result

    monkeypatch.setattr(runtime_module, "remove_agent_runtime_namespace", remove)
    monkeypatch.setattr(manager_module, "await_owned_task", join)
    # Declared already-completed shutdown input; this is actual offboarding
    # consumer/namespace proof, not full Agent shutdown or genesis acceptance.
    shutdown = create_owned_task(asyncio.sleep(0), retain_originals=True)
    await await_owned_task(shutdown)
    assert manager._handoff_shutdown_to_quarantined_reaper(
        name="OriginalOffboard", agent=agent, shutdown_task=shutdown,
        offboard_runtime=True,
    )
    reaper = next(iter(manager._quarantined_shutdown_reapers.values())).task
    try:
        await asyncio.wait_for(entered.wait(), 5)
        reaper.cancel("actual quarantine filesystem observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(reaper)
        if physical_failure:
            source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert not scope.path.exists() and reaper.done()
    finally:
        release.set()
        await await_owned_task(shutdown)
        await await_owned_task(reaper)
