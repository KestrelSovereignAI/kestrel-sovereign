"""Actual child-settlement withdrawal cannot be replaced with a new cancellation."""
import asyncio

import pytest

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["retirement", "reconciliation"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_child_termination_retains_actual_settlement_withdrawal(
    native_pg, onboarding_owner, monkeypatch, surface, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    actual_join, owner = ownership.await_owned_task, None
    async def observe(task, *args, **kwargs):
        result = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    parent = "did:test:settlement-parent"
    async def relations():
        return {agent.did: (parent, "Child")}
    async def remove(name, **kwargs):
        assert kwargs["offboard_runtime"] is False
        await db.execute("INSERT INTO host_rollback_effects VALUES (140)")
        raise source.error
    async def settlement(*args):
        await db.execute("INSERT INTO host_rollback_effects VALUES (141)")
        if surface == "retirement":
            entered.set()
            await release.wait()
    async def reconcile(*args):
        await db.execute("INSERT INTO host_rollback_effects VALUES (142)")
        if surface == "reconciliation":
            entered.set()
            await release.wait()
        return False
    monkeypatch.setattr(manager, "get_authoritative_spawn_relations", relations)
    monkeypatch.setattr(manager, "remove_agent", remove)
    monkeypatch.setattr(manager, "_settle_destructive_spawn_retirement", settlement)
    monkeypatch.setattr(manager, "_prune_child_tracking_if_fully_removed", reconcile)
    owner = create_owned_task(manager.terminate_child(parent, "Child"), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual child settlement observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 3
    finally:
        release.set()
        await await_owned_task(owner)
