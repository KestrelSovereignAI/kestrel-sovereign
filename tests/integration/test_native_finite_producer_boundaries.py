"""Real fleet/startup/renewal boundaries, with declared native failure seams.

No provider inference, deployed agents, financial refund or genesis acceptance.
"""
import asyncio
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from tests.fixtures.sdk_contribution_fixture import SDKFixtureHostFeature
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_execution_custody_review18_pg import setup_delivery, join_owned
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.integration.test_native_finite_outer_consumers import unstarted_native_close
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_single_agent_prior_continuation_and_later_shutdown_sources_stay_immutable(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    from kestrel_sovereign import server
    manager, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    resource, independent = await unstarted_native_close()
    async def continuation():
        await database.execute("INSERT INTO host_rollback_effects VALUES (61)")
        raise source.error
    async def shutdown():
        await database.execute("INSERT INTO host_rollback_effects VALUES (62)")
        raise independent.error
    # Actual KestrelAgent continuation waiter; shutdown callback is a declared
    # native-effect seam, not whole-agent teardown or provider acceptance.
    agent._durable_shutdown_continuation = create_owned_task(continuation(), retain_originals=True)
    monkeypatch.setattr(agent, "shutdown", shutdown)
    owner = create_owned_task(server._shutdown_single_agent(agent), retain_originals=True)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        independent.assert_retained(result.error)
        assert agent._durable_shutdown_continuation.done()
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects") == 2
    finally:
        await await_owned_task(owner)
        await await_owned_task(agent._durable_shutdown_continuation)
        await resource.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_fleet_sweep_preserves_two_sources_and_actual_lock_withdrawal(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    manager, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    resource, independent = await unstarted_native_close()
    # Declared removal callback, real SQLite effects, actual public fleet
    # sweep/lock acquisition and relationship reconciliation. Not agent shutdown.
    manager._agents = {"First": agent, "Second": agent}
    manager._parent_children = {agent.did: ["First", "Second"]}
    calls, withdrawals = [], []
    async def remove(name, **kwargs):
        calls.append(name)
        await database.execute("INSERT INTO host_rollback_effects VALUES (?)", (41 + len(calls),))
        manager._agents.pop(name)
        raise source.error if name == "First" else independent.error
    monkeypatch.setattr(manager, "remove_agent", remove)
    from kestrel_sovereign.multi_agent import agent_manager as module
    actual = module.await_lifecycle_task_completion
    async def observe(task, *args, **kwargs):
        result = await actual(task, *args, **kwargs)
        withdrawals[:] = kwargs.get("withdrawals") or withdrawals
        return result
    monkeypatch.setattr(module, "await_lifecycle_task_completion", observe)
    await manager._quarantined_shutdown_drain_lock.acquire()
    owner = create_owned_task(manager.shutdown_all(), retain_originals=True)
    try:
        await asyncio.sleep(0)
        owner.cancel("actual fleet drain lock observer withdrew")
        await asyncio.sleep(0)
        manager._quarantined_shutdown_drain_lock.release()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        independent.assert_retained(result.error)
        assert withdrawals and all(value in tuple(_explicit_control_chain(result.error)) for value in withdrawals)
        assert calls == ["First", "Second"]
        assert not manager._agents and not manager._parent_children and not manager._child_budgets
        assert not manager._quarantined_shutdown_drain_lock.locked()
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects") == 2
    finally:
        if manager._quarantined_shutdown_drain_lock.locked() and not owner.done():
            manager._quarantined_shutdown_drain_lock.release()
        await await_owned_task(owner)
        manager._agents.clear()
        await resource.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_startup_rejection_keeps_original_mount_and_later_hook_sources(
    native_pg, tmp_path, monkeypatch, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    resource, independent = await unstarted_native_close()
    config = tmp_path / "multi_agent.toml"
    config.write_text('[host]\nbind = "127.0.0.1"\nport = 8888\n')
    host_db = tmp_path / "host.db"
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "KESTREL_HOME": str(tmp_path), "PROJECT_DIR": str(tmp_path),
        "KESTREL_DB_PATH": str(tmp_path / "private-state"),
        "KESTREL_HOST_DB_PATH": str(host_db),
        "KESTREL_MULTI_AGENT_CONFIG": str(config), "KESTREL_MULTI_AGENT": "1",
        "KESTREL_API_KEY": "inert-native-host-auth-not-provider-credential",
        "KESTREL_DB_BACKEND": "sqlite", "KESTREL_HOLD_BACKEND": "sqlite",
        "KESTREL_PHOENIX_ENABLED": "0", "KESTREL_TRACING_ENABLED": "0",
    }.items():
        monkeypatch.setenv(key, value)
    from kestrel_sovereign import host_features, server
    contexts, calls = [], []
    class FailingStop(SDKFixtureHostFeature):
        async def on_host_start(self, ctx):
            await super().on_host_start(ctx)
            contexts.append(ctx)
            await ctx.db.execute("CREATE TABLE effects (id INTEGER)")
            await ctx.db.execute("INSERT INTO effects VALUES (1)")
        async def on_host_stop(self, ctx):
            calls.append("stop")
            await ctx.db.execute("INSERT INTO effects VALUES (2)")
            raise independent.error
    feature = FailingStop()
    monkeypatch.setattr(host_features, "instantiate_host_features", lambda **kwargs: [feature])
    actual_mount = host_features.mount_host_feature_routers
    def mount(*args, **kwargs):
        actual_mount(*args, **kwargs)
        calls.append("mount")
        raise source.error
    monkeypatch.setattr(host_features, "mount_host_feature_routers", mount)
    app = FastAPI()
    async def invoke():
        async with server._lifespan_startup(app):
            pytest.fail("unsafe startup published")
    owner = create_owned_task(invoke(), retain_originals=True)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        independent.assert_retained(result.error)
        assert calls == ["mount", "stop"]
        assert len(contexts) == 1 and app.state.agent_manager.list_agents() == {}
        assert not app.state.host_context_publication_gate.is_set()
        with sqlite3.connect(host_db) as observer:
            assert observer.execute("SELECT count(*) FROM effects").fetchone() == (2,)
    finally:
        await await_owned_task(owner)
        # Retained failed hook remains evidence; every independent native owner
        # must still physically close under the actual outer server drain.
        await server._shutdown_server_resources(app)
        for ctx in contexts:
            assert ctx.db.backend._connection is None
        await resource.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_lease_context_finalizer_preserves_body_source_and_real_join_cancellation(
    native_pg, monkeypatch, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    resource, independent = await unstarted_native_close()
    agent, dispatcher, delivery, _ = await setup_delivery(native_pg)
    delivery = replace(delivery, lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=0.03))
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    actual_renew = dispatcher.renew_durable_delivery_lease
    actual_join = ownership.await_owned_task
    owner = None
    async def renew(**kwargs):
        renewed = await actual_renew(**kwargs)
        assert renewed is not None
        entered.set()
        await release.wait()
        raise independent.error
    async def observe(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome
    monkeypatch.setattr(dispatcher, "renew_durable_delivery_lease", renew)
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    async def invoke():
        async with dispatcher._renew_durable_cognition_lease(delivery):
            await entered.wait()
            raise source.error
    owner = create_owned_task(invoke(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.sleep(0)
        owner.cancel("actual renewal finalizer observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        independent.assert_retained(result.error)
        assert withdrawals and all(value in tuple(_explicit_control_chain(result.error)) for value in withdrawals)
        row = await native_pg[1].fetchrow("SELECT status, lease_token FROM durable_signal_deliveries WHERE delivery_id='delivery1'")
        assert row["status"] == "leased" and row["lease_token"] == delivery.lease_token
        assert not any(task.get_name().startswith("durable_cognition_lease_renewal:") and not task.done() for task in asyncio.all_tasks())
    finally:
        release.set()
        await await_owned_task(owner)
        await join_owned(agent, dispatcher)
        await resource.close()
