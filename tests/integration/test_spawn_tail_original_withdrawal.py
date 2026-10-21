"""Spawn tail projections keep actual native sources and finite withdrawals."""
import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_terminal_error
from kestrel_sovereign.multi_agent import agent_manager as manager_module
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["slot", "name", "release-error", "direct-release-error"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_spawn_retirement_retains_body_and_actual_observer_evidence(
    native_pg, onboarding_owner, monkeypatch, surface, source_cancel,
):
    # The tail owns real manager admissions/cap state. The body is a declared
    # failing spawn seam, not a claim of signed genesis/provider acceptance.
    manager, parent, _database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    release_error = OSError("independent spawn name retirement observation failed")
    actual_join, actual_release = manager_module.await_owned_task, manager._release_agent_operation
    actual_direct = manager._do_spawn

    async def join(*args, **kwargs):
        result = await actual_join(*args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result

    async def body(*_args, **_kwargs):
        if surface == "slot":
            await manager._lock.acquire()
            entered.set()
        raise source.error

    async def retire(admission):
        if surface != "slot":
            await manager._lock.acquire()
            entered.set()
        try:
            result = await actual_release(admission)
        except BaseException as error:
            if surface.endswith("release-error"):
                raise execution_terminal_error(error, release_error,
                    retain_originals=True, isolate_control_originals=True)
            raise
        if surface.endswith("release-error"):
            raise release_error
        return result

    monkeypatch.setattr(manager_module, "await_owned_task", join)
    monkeypatch.setattr(manager, "_do_spawn", body)
    monkeypatch.setattr(manager, "_release_agent_operation", retire)
    mandate = SimpleNamespace(max_child_depth=0)
    if surface == "direct-release-error":
        operation = actual_direct("TailChild", parent, mandate)
    else:
        admission, created = await manager._admit_agent_operation("TailChild", kind="spawn")
        assert created
        operation = manager._run_admitted_spawn("TailChild", parent, mandate, admission)
    owner = create_owned_task(operation, retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        # Drain the original release/slot Task, not a substitute cleanup Task.
        owner.cancel("actual spawn-tail observer withdrew")
        await asyncio.sleep(0)
        manager._lock.release()
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        chain = tuple(_explicit_control_chain(result.error))
        assert withdrawals and all(error in chain for error in withdrawals)
        if surface.endswith("release-error"):
            assert release_error in chain
        assert not manager._agent_operations and manager._pending_spawns == 0
        if surface != "direct-release-error":
            assert not admission.spawn_slot_active and admission.spawn_slot_terminal.done()
            assert vars(admission)["_spawn_slot_retirement_owner"].admission.task.done()
    finally:
        release.set()
        if manager._lock.locked():
            manager._lock.release()
        await await_owned_task(owner)
