"""Finite host cleanup keeps the actual C, not a fresh cancellation lookalike."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.multi_agent import agent_manager as manager_module
from kestrel_sovereign.multi_agent.config import LocalAgentConfig
from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
async def test_native_onboarding_retains_actual_successful_cleanup_observer_withdrawal(
    native_pg, onboarding_owner, tmp_path, monkeypatch,
):
    manager, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    actual_join = manager_module.await_owned_task
    async def join(*args, **kwargs):
        result = await actual_join(*args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    async def cleanup():
        await database.execute("INSERT INTO host_rollback_effects VALUES (286)")
        entered.set()
        await release.wait()
    async def before(_agent):
        raise source.error
    async def initialize(*args, **kwargs):
        return agent
    actual_admit = manager._admit_agent_operation
    async def admit(*args, **kwargs):
        admission, owns = await actual_admit(*args, **kwargs)
        admission.before_publish = before
        return admission, owns
    monkeypatch.setattr(manager_module, "await_owned_task", join)
    monkeypatch.setattr(manager, "_initialize_agent", initialize)
    monkeypatch.setattr(manager, "_admit_agent_operation", admit)
    monkeypatch.setattr(agent, "shutdown", cleanup)
    owner = create_owned_task(manager.load_agent("Original", LocalAgentConfig(data_dir=tmp_path / "host", port=8866)), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual unpublished cleanup observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=286") == 1
    finally:
        release.set()
        await await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("startup", [False, True])
async def test_native_activation_keeps_withdrawal_inside_ordinary_cleanup_composition(
    native_pg, onboarding_owner, monkeypatch, startup,
):
    _, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    ordinary = OSError("foreign on_disable prose must not enter SafeMode")
    class Feature(SDKFixtureFeature):
        contribution_prefix = "enclosed-withdrawal-207"
        async def on_enable(self):
            raise source.error
        async def on_disable(self):
            raise ordinary
        async def shutdown(self):
            await database.execute("INSERT INTO host_rollback_effects VALUES (288)")
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError as error:
                withdrawals.append(error)
                raise
    feature = Feature(agent)
    operation = agent._shutdown_failed_feature(feature, opening=source.error) if startup else agent._activate_feature_runtime(feature)
    owner = create_owned_task(operation, retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual enclosed canonical cleanup withdrawal")
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        chain = tuple(_explicit_control_chain(result.error))
        assert withdrawals and all(error in chain for error in withdrawals)
        assert ordinary not in chain, "foreign cleanup prose escaped SafeMode projection"
        assert agent._feature_lifecycle_integrity_uncertain and agent._safe_mode
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=288") == 1
    finally:
        release.set()
        await await_owned_task(owner)


@pytest.mark.asyncio
async def test_native_onboarding_keeps_actual_final_name_release_withdrawal(
    native_pg, onboarding_owner, tmp_path, monkeypatch,
):
    manager, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    entered, withdrawals = asyncio.Event(), []
    actual_release, actual_join = manager._release_agent_operation, manager_module.await_owned_task
    async def join(*args, **kwargs):
        result = await actual_join(*args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    async def held_release(admission):
        await manager._lock.acquire()
        entered.set()
        return await actual_release(admission)
    async def initialize(*args, **kwargs):
        return agent
    async def cleanup():
        await database.execute("INSERT INTO host_rollback_effects VALUES (289)")
    async def before(_agent):
        raise source.error
    actual_admit = manager._admit_agent_operation
    async def admit(*args, **kwargs):
        admission, owns = await actual_admit(*args, **kwargs)
        admission.before_publish = before
        return admission, owns
    monkeypatch.setattr(manager_module, "await_owned_task", join)
    monkeypatch.setattr(manager, "_release_agent_operation", held_release)
    monkeypatch.setattr(manager, "_admit_agent_operation", admit)
    monkeypatch.setattr(manager, "_initialize_agent", initialize)
    monkeypatch.setattr(agent, "shutdown", cleanup)
    owner = create_owned_task(manager.load_agent("Original", LocalAgentConfig(data_dir=tmp_path / "held-name", port=8867)), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual final name release observer withdrew")
        await asyncio.sleep(0)
        manager._lock.release()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=289") == 1
        assert not manager._agent_operations
    finally:
        if manager._lock.locked():
            manager._lock.release()
        await await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_opening", [False, True])
async def test_activation_native_failure_preserves_actual_independent_withdrawal(
    native_pg, onboarding_owner, monkeypatch, cancel_opening,
):
    _, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    async def wait_for_withdrawal():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError as error:
            withdrawals.append(error)
            raise
    class Feature(SDKFixtureFeature):
        contribution_prefix = "actual-activation-withdrawal-206"
        async def on_enable(self):
            if cancel_opening:
                await wait_for_withdrawal()
            raise source.error
        async def on_disable(self):
            await database.execute("INSERT INTO host_rollback_effects VALUES (287)")
            if not cancel_opening:
                await wait_for_withdrawal()
            else:
                raise source.error
    feature = Feature(agent)
    owner = create_owned_task(agent._activate_feature_runtime(feature), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual runtime activation withdrew")
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert agent._feature_lifecycle_integrity_uncertain and agent._safe_mode
    finally:
        release.set()
        await await_owned_task(owner)
