"""Early fleet exits and ordinary-first single-agent shutdown keep custody.

Callbacks model component removal/pruning/continuation with real SQLite effects.
They do not grant signed authority, close deployed agents, refund funds or call providers.
"""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.spawn.mandate import SpawnMandate
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("group", [False, True])
async def test_native_topology_cancellation_does_not_skip_independent_fleet_cleanup(
    native_pg, onboarding_owner, monkeypatch, group,
):
    manager, agent, database = onboarding_owner
    original = await _original_control(native_pg[0], monkeypatch, True)
    source = OriginalFailure.capture(BaseExceptionGroup("genuine topology read source", [original, OSError("same source diagnosis")]) if group else original)
    calls = []
    async def topology():
        assert await native_pg[1].fetchval("SELECT 1") == 1
        raise source.error
    async def remove(name, **kwargs):
        assert kwargs["offboard_runtime"] is False
        calls.append(name)
        await database.execute("INSERT INTO host_rollback_effects VALUES (?)", (80 + len(calls),))
        manager._agents.pop(name)
        return True
    manager._agents = {"First": agent, "Second": agent}
    monkeypatch.setattr(manager, "_verified_spawn_relations", topology)
    monkeypatch.setattr(manager, "remove_agent", remove)
    owner = create_owned_task(manager.shutdown_all(), retain_originals=True)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert calls == ["First", "Second"] and not manager._agents
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects") == 2
        assert not manager._quarantined_shutdown_drain_lock.locked()
    finally:
        await await_owned_task(owner)
        manager._agents.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_postremoval_pruning_withdrawal_keeps_prior_native_error_and_sweeps_peer(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    manager, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    calls, withdrawals = [], []
    manager._agents = {"First": agent, "Second": agent}
    # Uncommitted tracking metadata, explicitly NOT an authority grant.
    mandate = SpawnMandate(parent_did=agent.did, child_did=agent.did, authority_committed=False)
    manager._child_mandates["First"] = mandate
    manager._parent_children[agent.did] = ["First", "Second"]
    async def remove(name, **kwargs):
        calls.append(name)
        if name == "Second":
            assert manager._child_mandates.get("First") is mandate
            assert "First" in manager._parent_children[agent.did]
        await database.execute("INSERT INTO host_rollback_effects VALUES (?)", (83 + len(calls),))
        manager._agents.pop(name)
        if name == "First":
            raise source.error
        return True
    async def prune(parent, child):
        assert child == "First" and parent == agent.did
        await database.execute("INSERT INTO host_rollback_effects VALUES (86)")
        asyncio.current_task().cancel("actual fleet pruning observer withdrew")
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as error:
            withdrawals.append(error)
            raise
    monkeypatch.setattr(manager, "remove_agent", remove)
    monkeypatch.setattr(manager, "_prune_child_tracking_if_fully_removed", prune)
    owner = create_owned_task(manager.shutdown_all(), retain_originals=True)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert calls == ["First", "Second"] and not manager._agents
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects") == 3
        assert not manager._quarantined_shutdown_drain_lock.locked()
    finally:
        await await_owned_task(owner)
        manager._agents.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_single_agent_keeps_ordinary_prior_and_current_c_before_native_waiter(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    from kestrel_sovereign import server
    _, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    ordinary = OriginalFailure.capture(OSError("original ordinary prior continuation"))
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    calls, owner = [], None
    actual_wait_for = asyncio.wait_for
    async def observe(awaitable, *args, **kwargs):
        try:
            return await actual_wait_for(awaitable, *args, **kwargs)
        except asyncio.CancelledError as error:
            if asyncio.current_task() is owner:
                withdrawals.append(error)
            raise
    async def completed():
        await database.execute("INSERT INTO host_rollback_effects VALUES (90)")
    continuation = create_owned_task(completed(), retain_originals=True)
    await await_owned_task(continuation)
    agent._durable_shutdown_continuation = continuation
    async def waiter():
        calls.append(asyncio.current_task())
        await database.execute("INSERT INTO host_rollback_effects VALUES (?)", (90 + len(calls),))
        raise ordinary.error if len(calls) == 1 else source.error
    async def shutdown():
        entered.set()
        try:
            await release.wait()
        finally:
            await database.execute("INSERT INTO host_rollback_effects VALUES (93)")
    monkeypatch.setattr(agent, "wait_for_shutdown_completion", waiter)
    monkeypatch.setattr(agent, "shutdown", shutdown)
    monkeypatch.setattr(asyncio, "wait_for", observe)
    owner = create_owned_task(server._shutdown_single_agent(agent), retain_originals=True)
    try:
        await actual_wait_for(entered.wait(), 5)
        owner.cancel("actual single-agent shutdown observer withdrew")
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        ordinary.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert len(calls) == 2 and all(task.done() for task in calls)
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects") == 4
    finally:
        release.set()
        await await_owned_task(owner)
        await await_owned_task(continuation)
