"""Unit tests for SpawnedAgentLifecycle and hook events."""

import asyncio
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kestrel_sdk.hooks.base import HookEvent, HookInput, HookOutput

from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.inception_service import generate_secp256k1_keypair
from kestrel_sovereign.multi_agent.agent_manager import (
    AgentManager,
    RuntimeOffboardingRetainedError,
)
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.spawn.authority_registry import SpawnAuthorityRegistry
from kestrel_sovereign.spawn.lifecycle import (
    SpawnedAgentLifecycle,
    SpawnMode,
    SpawnResult,
    SpawnStatus,
    _terminal_retirement_tree,
)
from kestrel_sovereign.spawn.mandate import (
    PersistedSpawnMandateExpiredError,
    SpawnMandate,
    sign_mandate,
)


def _make_mock_manager():
    """Create a mock AgentManager with terminate_child."""
    manager = MagicMock()
    manager.terminate_child = AsyncMock(return_value=True)
    manager.get_children = MagicMock(return_value=[])
    manager.get_agent = MagicMock(return_value=None)
    return manager


def test_restored_ephemeral_ttl_rearms_after_sync_construction() -> None:
    """A lifecycle first built without a loop must arm its timer later."""

    manager = AgentManager()
    mandate = SpawnMandate(
        parent_did="did:test:parent",
        child_did="did:test:child",
        ttl_seconds=3600,
    )
    manager._child_mandates["Restored"] = mandate
    lifecycle = SpawnedAgentLifecycle(manager)
    assert lifecycle._tracked["Restored"].ttl_task is None

    async def rearm() -> None:
        lifecycle.restore_from_manager()
        ttl_task = lifecycle._tracked["Restored"].ttl_task
        assert ttl_task is not None
        lifecycle.withdraw_persisted_child("Restored")
        await asyncio.sleep(0)
        assert ttl_task.cancelled()

    asyncio.run(rearm())


@pytest.mark.asyncio
async def test_feature_registration_preserves_manager_armed_signed_timer() -> None:
    """The hook-registration seam cannot reset a TTL already transferred."""

    manager = _make_mock_manager()
    lifecycle = SpawnedAgentLifecycle(manager)
    mandate = SpawnMandate(
        parent_did="did:test:timer-parent",
        child_did="did:test:timer-child",
        ttl_seconds=3600,
    )
    lifecycle.restore_persisted_child("TimerChild", mandate, arm_ttl=True)
    manager_timer = lifecycle._tracked["TimerChild"].ttl_task

    await lifecycle.register(
        "TimerChild",
        mandate.child_did,
        mandate.parent_did,
        ttl_seconds=mandate.ttl_seconds,
        mode=SpawnMode.EPHEMERAL,
        purpose="registered after governance commit",
        started_at=mandate.created_at,
    )

    assert lifecycle._tracked["TimerChild"].ttl_task is manager_timer
    lifecycle.withdraw_persisted_child("TimerChild")


@pytest.mark.asyncio
async def test_failed_publication_returns_transferred_timer_to_cold_authority(
    monkeypatch, tmp_path
) -> None:
    """A late load failure cannot cancel the host witness's deadline."""

    child_name = "FailedPublicationChild"
    child_did = "did:test:failed-publication-child"
    parent_did = "did:test:failed-publication-parent"
    mandate = SpawnMandate(
        parent_did=parent_did,
        child_did=child_did,
        ttl_seconds=1,
        parent_signature="signed-failed-publication",
    )
    registry = SpawnAuthorityRegistry(tmp_path)
    registry.record_active(
        child_name=child_name,
        child_did=child_did,
        mandate=mandate,
        config=LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        ),
    )
    manager = AgentManager(base_data_dir=tmp_path)
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    monkeypatch.setattr(
        lifecycle,
        "_remaining_ttl_seconds",
        lambda _created_at, _ttl_seconds: 0.01,
    )
    lifecycle.arm_cold_authority_ttl(child_name, mandate)
    cold_owner = lifecycle._cold_ttl_tasks[(child_name.casefold(), child_did)]
    lifecycle.restore_persisted_child(child_name, mandate, arm_ttl=False)
    lifecycle.arm_restored_child_ttl(
        child_name,
        expected_child_did=child_did,
    )
    assert lifecycle._tracked[child_name].ttl_task is cold_owner

    lifecycle.withdraw_persisted_child(
        child_name,
        expected_child_did=child_did,
        preserve_ttl_owner=True,
    )

    assert not cold_owner.cancelled()
    await asyncio.wait_for(asyncio.shield(cold_owner), timeout=0.5)
    assert registry.get(child_did).retired


@pytest.mark.asyncio
async def test_transferred_cold_ttl_uses_case_preserved_routing_name(
    monkeypatch, tmp_path
) -> None:
    """A registry spelling cannot strand expiry beside a case-variant route."""

    registry_name = "CaseVariantChild"
    route_name = "casevariantchild"
    child_did = "did:test:case-variant-child"
    parent_did = "did:test:case-variant-parent"
    mandate = SpawnMandate(
        parent_did=parent_did,
        child_did=child_did,
        ttl_seconds=1,
        parent_signature="signed-case-variant-child",
    )
    config = LocalAgentConfig(
        data_dir=Path("agent_data") / registry_name,
        port=8802,
    )
    registry = SpawnAuthorityRegistry(tmp_path)
    registry.record_active(
        child_name=registry_name,
        child_did=child_did,
        mandate=mandate,
        config=config,
    )
    manager = AgentManager(base_data_dir=tmp_path)
    child = MagicMock(agent_id=child_did)
    child.agent_id = child_did
    child.shutdown = AsyncMock()
    manager._agents[route_name] = child
    manager._agent_names[child_did] = route_name
    manager._parent_children[parent_did] = [route_name]
    manager._child_mandates[route_name] = mandate
    manager._created_configs[route_name] = config
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    lifecycle.withdraw_persisted_child(route_name)
    monkeypatch.setattr(
        lifecycle,
        "_remaining_ttl_seconds",
        lambda _created_at, _ttl_seconds: 0.01,
    )

    lifecycle.arm_cold_authority_ttl(registry_name, mandate)
    lifecycle.restore_persisted_child(
        route_name,
        mandate,
        authority_parent_did=parent_did,
        arm_ttl=False,
    )
    lifecycle.arm_restored_child_ttl(route_name, expected_child_did=child_did)
    owner = lifecycle._tracked[route_name].ttl_task

    try:
        for _ in range(100):
            witness = registry.get(child_did)
            if witness is not None and witness.retired:
                break
            await asyncio.sleep(0.01)

        assert registry.get(child_did).retired
        assert manager.get_agent(route_name) is None
        child.shutdown.assert_awaited_once()
    finally:
        if owner is not None and not owner.done():
            owner.cancel()


@pytest.mark.asyncio
async def test_expired_live_child_stays_denied_when_shutdown_is_refused(
    tmp_path,
) -> None:
    """Expiry is a routing/authority fence even while cleanup remains retryable."""

    child_name = "RefusedExpiredChild"
    child_did = "did:test:refused-expired-child"
    parent_did = "did:test:refused-expired-parent"
    mandate = SpawnMandate(
        parent_did=parent_did,
        child_did=child_did,
        ttl_seconds=1,
        parent_signature="signed-refused-expired-child",
    )
    config = LocalAgentConfig(
        data_dir=Path("agent_data") / child_name,
        port=8802,
    )
    registry = SpawnAuthorityRegistry(tmp_path)
    registry.record_active(
        child_name=child_name,
        child_did=child_did,
        mandate=mandate,
        config=config,
    )
    manager = AgentManager(base_data_dir=tmp_path)
    child = MagicMock(agent_id=child_did)
    child.agent_id = child_did
    child.shutdown = AsyncMock(side_effect=RuntimeError("cleanup refused"))
    manager._agents[child_name] = child
    manager._agent_names[child_did] = child_name
    manager._parent_children[parent_did] = [child_name]
    manager._child_mandates[child_name] = mandate
    manager._created_configs[child_name] = config
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle

    terminated = await lifecycle._terminate_and_cleanup(
        child_name,
        SpawnStatus.TIMED_OUT,
        reason="TTL expired",
    )

    assert terminated is False
    assert manager.get_agent(child_name) is None
    assert manager._get_agent_for_lifecycle(child_name) is child
    assert registry.get(child_did).state == "retiring"
    assert lifecycle.is_tracked(child_name)


@pytest.mark.asyncio
async def test_cold_authority_expiry_revokes_live_scheduler_scope(
    monkeypatch, tmp_path
) -> None:
    """A retired cold witness cannot remain scheduler-authorized in-process."""

    child_name = "ExpiredColdSchedulerChild"
    child_did = "did:test:expired-cold-scheduler-child"
    mandate = SpawnMandate(
        parent_did="did:test:expired-cold-scheduler-parent",
        child_did=child_did,
        ttl_seconds=1,
        parent_signature="signed-expired-cold-scheduler",
    )
    config = LocalAgentConfig(
        data_dir=Path("agent_data") / child_name,
        port=8802,
        autostart=False,
    )
    registry = SpawnAuthorityRegistry(tmp_path)
    registry.record_active(
        child_name=child_name,
        child_did=child_did,
        mandate=mandate,
        config=config,
    )
    manager = AgentManager(base_data_dir=tmp_path)
    manager._seed_scheduler_authority({child_did: (child_name, config)})
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    monkeypatch.setattr(
        lifecycle,
        "_remaining_ttl_seconds",
        lambda _created_at, _ttl_seconds: 0.01,
    )

    execution_lease = manager.scheduler_execution_lease(child_did)
    await execution_lease.__aenter__()
    lifecycle.arm_cold_authority_ttl(child_name, mandate)
    owner = lifecycle._cold_ttl_tasks[(child_name.casefold(), child_did)]
    await asyncio.sleep(0.05)
    assert not owner.done()
    assert manager.is_scheduler_agent_authorized(child_did)
    await execution_lease.__aexit__(None, None, None)
    await asyncio.wait_for(asyncio.shield(owner), timeout=0.5)

    assert registry.get(child_did).retired
    assert not manager.is_scheduler_agent_authorized(child_did)
    assert manager.scheduler_authority_for(child_did) is None


@pytest.mark.asyncio
async def test_terminal_scheduler_revocation_preserves_same_name_replacement(
    tmp_path,
) -> None:
    """Terminal identity cleanup cannot revoke a replacement by display name."""

    name = "ReusedSchedulerName"
    old_did = "did:test:expired-scheduler-identity"
    replacement_did = "did:test:replacement-scheduler-identity"
    old_config = LocalAgentConfig(data_dir="agent_data/old", port=8802)
    replacement_config = LocalAgentConfig(
        data_dir="agent_data/replacement",
        port=8803,
    )
    manager = AgentManager(base_data_dir=tmp_path)
    manager._seed_scheduler_authority(
        {
            old_did: (name, old_config),
            replacement_did: (name, replacement_config),
        }
    )

    await manager.revoke_terminal_spawn_scheduler_authority(name, old_did)

    assert not manager.is_scheduler_agent_authorized(old_did)
    assert manager.scheduler_authority_for(old_did) is None
    assert manager.is_scheduler_agent_authorized(replacement_did)
    assert manager.scheduler_authority_for(replacement_did) == (
        name,
        replacement_config,
    )
    assert manager._scheduler_authority_by_name[name] == replacement_did
    assert name not in manager._scheduler_revoked_names


@pytest.mark.asyncio
async def test_failed_cold_authority_retirement_keeps_owner_and_retries(
    monkeypatch, tmp_path
) -> None:
    """A transient retirement failure cannot permanently orphan its witness."""

    child_name = "RetryColdRetirementChild"
    child_did = "did:test:retry-cold-retirement-child"
    mandate = SpawnMandate(
        parent_did="did:test:retry-cold-retirement-parent",
        child_did=child_did,
        ttl_seconds=1,
        parent_signature="signed-retry-cold-retirement",
    )
    config = LocalAgentConfig(
        data_dir=Path("agent_data") / child_name,
        port=8802,
        autostart=False,
    )
    registry = SpawnAuthorityRegistry(tmp_path)
    registry.record_active(
        child_name=child_name,
        child_did=child_did,
        mandate=mandate,
        config=config,
    )
    manager = AgentManager(base_data_dir=tmp_path)
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    monkeypatch.setattr(
        lifecycle,
        "_remaining_ttl_seconds",
        lambda _created_at, _ttl_seconds: 0.01,
    )
    monkeypatch.setattr(
        "kestrel_sovereign.spawn.lifecycle._COLD_TTL_RETIREMENT_RETRY_SECONDS",
        0.01,
        raising=False,
    )
    original_record = manager.record_expired_spawn_retirement
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient registry write failure")
        return original_record(*args, **kwargs)

    manager.record_expired_spawn_retirement = fail_once
    lifecycle.arm_cold_authority_ttl(child_name, mandate)
    key = (child_name.casefold(), child_did)
    first_owner = lifecycle._cold_ttl_tasks[key]
    await asyncio.wait_for(asyncio.shield(first_owner), timeout=0.5)

    for _ in range(100):
        if registry.get(child_did).retired and key not in lifecycle._cold_ttl_tasks:
            break
        await asyncio.sleep(0.01)

    assert attempts == 2
    assert registry.get(child_did).retired
    assert key not in lifecycle._cold_ttl_tasks


def test_terminal_retirement_retry_keeps_retiring_descendants(tmp_path) -> None:
    """A retry must settle descendants already durably marked retiring."""

    root_did = "did:test:retirement-retry-root"
    parent_name = "RetirementRetryParent"
    parent_did = "did:test:retirement-retry-parent"
    child_name = "RetirementRetryChild"
    child_did = "did:test:retirement-retry-child"
    registry = SpawnAuthorityRegistry(tmp_path)
    for name, did, authority_parent, port in (
        (parent_name, parent_did, root_did, 8802),
        (child_name, child_did, parent_did, 8803),
    ):
        registry.record_active(
            child_name=name,
            child_did=did,
            mandate=SpawnMandate(
                parent_did=authority_parent,
                child_did=did,
                parent_signature=f"signed-{name}",
            ),
            config=LocalAgentConfig(
                data_dir=Path("agent_data") / name,
                port=port,
                autostart=False,
            ),
        )

    manager = AgentManager(base_data_dir=tmp_path)
    manager.begin_terminal_spawn_retirements(
        ((parent_name, parent_did), (child_name, child_did))
    )

    targets = _terminal_retirement_tree(
        manager,
        child_name=parent_name,
        child_did=parent_did,
        parent_did=root_did,
    )

    assert [(target.child_name, target.child_did) for target in targets] == [
        (parent_name, parent_did),
        (child_name, child_did),
    ]


@pytest.mark.asyncio
async def test_live_ttl_retirement_failure_retains_custody_and_retries(
    monkeypatch, tmp_path
) -> None:
    """A removed live child keeps an owner until its witness is retired."""

    child_name = "RetryLiveRetirementChild"
    child_did = "did:test:retry-live-retirement-child"
    parent_did = "did:test:retry-live-retirement-parent"
    mandate = SpawnMandate(
        parent_did=parent_did,
        child_did=child_did,
        ttl_seconds=1,
        parent_signature="signed-retry-live-retirement",
    )
    registry = SpawnAuthorityRegistry(tmp_path)
    registry.record_active(
        child_name=child_name,
        child_did=child_did,
        mandate=mandate,
        config=LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        ),
    )
    manager = AgentManager(base_data_dir=tmp_path)
    child = MagicMock(agent_id=child_did)
    child.agent_id = child_did
    manager._agents[child_name] = child
    manager._agent_names[child_did] = child_name
    manager._parent_children[parent_did] = [child_name]
    manager._child_mandates[child_name] = mandate
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    monkeypatch.setattr(
        lifecycle,
        "_remaining_ttl_seconds",
        lambda _created_at, _ttl_seconds: 0.01,
    )
    monkeypatch.setattr(
        "kestrel_sovereign.spawn.lifecycle._COLD_TTL_RETIREMENT_RETRY_SECONDS",
        0.01,
        raising=False,
    )

    async def remove_once(_parent_did, _child_name):
        if manager._agents.pop(child_name, None) is None:
            return False
        manager._agent_names.pop(child_did, None)
        manager._parent_children[parent_did].remove(child_name)
        manager._child_mandates.pop(child_name, None)
        return True

    manager.terminate_child = AsyncMock(side_effect=remove_once)
    original_record = manager.record_expired_spawn_retirement
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient live retirement write failure")
        return original_record(*args, **kwargs)

    manager.record_expired_spawn_retirement = fail_once
    await lifecycle.register(
        child_name=child_name,
        child_did=child_did,
        parent_did=parent_did,
        ttl_seconds=1,
        mode=SpawnMode.EPHEMERAL,
        started_at=mandate.created_at,
    )
    first_owner = lifecycle._tracked[child_name].ttl_task
    assert first_owner is not None
    await asyncio.wait_for(asyncio.shield(first_owner), timeout=0.5)

    for _ in range(100):
        if registry.get(child_did).retired and not lifecycle.is_tracked(child_name):
            break
        await asyncio.sleep(0.01)

    assert attempts == 2
    assert registry.get(child_did).retired
    assert not lifecycle.is_tracked(child_name)


@pytest.mark.asyncio
async def test_direct_stop_preserves_finite_authority_expiry_owner(
    monkeypatch, tmp_path
) -> None:
    """An ordinary DELETE cannot turn a finite witness into a permanent slot."""

    child_name = "StoppedFiniteChild"
    child_did = "did:test:stopped-finite-child"
    parent_did = "did:test:stopped-finite-parent"
    mandate = SpawnMandate(
        parent_did=parent_did,
        child_did=child_did,
        ttl_seconds=1,
        parent_signature="signed-stopped-finite",
    )
    registry = SpawnAuthorityRegistry(tmp_path)
    registry.record_active(
        child_name=child_name,
        child_did=child_did,
        mandate=mandate,
        config=LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        ),
    )
    manager = AgentManager(base_data_dir=tmp_path)
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    monkeypatch.setattr(
        lifecycle,
        "_remaining_ttl_seconds",
        lambda _created_at, _ttl_seconds: 0.01,
    )
    manager._parent_children[parent_did] = [child_name]
    manager._child_mandates[child_name] = mandate
    manager._created_configs[child_name] = LocalAgentConfig(
        data_dir=Path("agent_data") / child_name,
        port=8802,
    )
    lifecycle.restore_persisted_child(child_name, mandate)
    expiry_owner = lifecycle._tracked[child_name].ttl_task
    child = MagicMock()
    child.agent_id = child_did
    child.did = child_did
    child.shutdown = AsyncMock()
    manager._agents[child_name] = child
    manager._agent_names[child_did] = child_name

    assert await manager.remove_agent(child_name) is True

    cold_key = (child_name.casefold(), child_did)
    assert lifecycle._cold_ttl_tasks[cold_key] is expiry_owner
    assert not expiry_owner.cancelled()
    await asyncio.wait_for(asyncio.shield(expiry_owner), timeout=0.5)
    assert registry.get(child_did).retired


@pytest.mark.asyncio
async def test_directly_stopped_parent_expiry_terminates_live_descendant(
    monkeypatch, tmp_path
) -> None:
    """An expired stopped parent cannot leave a live child under dead authority."""

    root_did = "did:test:direct-stop-root"
    parent_name = "DirectStoppedParent"
    parent_did = "did:test:direct-stopped-parent"
    child_name = "LiveDescendant"
    child_did = "did:test:live-descendant"
    parent_mandate = SpawnMandate(
        parent_did=root_did,
        child_did=parent_did,
        ttl_seconds=1,
        parent_signature="signed-direct-stopped-parent",
    )
    child_mandate = SpawnMandate(
        parent_did=parent_did,
        child_did=child_did,
        ttl_seconds=3600,
        parent_signature="signed-live-descendant",
    )
    registry = SpawnAuthorityRegistry(tmp_path)
    for name, did, mandate, port in (
        (parent_name, parent_did, parent_mandate, 8802),
        (child_name, child_did, child_mandate, 8803),
    ):
        registry.record_active(
            child_name=name,
            child_did=did,
            mandate=mandate,
            config=LocalAgentConfig(
                data_dir=Path("agent_data") / name,
                port=port,
            ),
        )

    manager = AgentManager(base_data_dir=tmp_path)
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    monkeypatch.setattr(
        lifecycle,
        "_remaining_ttl_seconds",
        lambda _created_at, _ttl_seconds: 0.01,
    )
    parent = MagicMock(agent_id=parent_did)
    parent.agent_id = parent_did
    parent.shutdown = AsyncMock()
    child = MagicMock(agent_id=child_did)
    child.agent_id = child_did
    child.shutdown = AsyncMock()
    manager._agents.update({parent_name: parent, child_name: child})
    manager._agent_names.update({parent_did: parent_name, child_did: child_name})
    manager._parent_children.update(
        {root_did: [parent_name], parent_did: [child_name]}
    )
    manager._child_mandates.update(
        {parent_name: parent_mandate, child_name: child_mandate}
    )
    lifecycle.restore_persisted_child(parent_name, parent_mandate)
    expiry_owner = lifecycle._tracked[parent_name].ttl_task

    assert await manager.remove_agent(parent_name) is True
    assert manager.get_agent(child_name) is child

    assert expiry_owner is not None
    await asyncio.wait_for(asyncio.shield(expiry_owner), timeout=0.5)
    child.shutdown.assert_awaited_once()
    assert manager.get_agent(child_name) is None
    assert registry.get(parent_did).retired
    assert registry.get(child_did).retired
    assert manager._spawn_cap_slots_in_use() == 0


@pytest.mark.asyncio
async def test_direct_terminal_prune_cancels_finite_authority_expiry_owner(
    tmp_path,
) -> None:
    """A retirement intent must not leave a redundant cold deadline owner."""

    child_name = "TerminalFiniteChild"
    child_did = "did:test:terminal-finite-child"
    mandate = SpawnMandate(
        parent_did="did:test:terminal-finite-parent",
        child_did=child_did,
        ttl_seconds=3600,
        parent_signature="signed-terminal-finite",
    )
    registry = SpawnAuthorityRegistry(tmp_path)
    registry.record_active(
        child_name=child_name,
        child_did=child_did,
        mandate=mandate,
        config=LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        ),
    )
    manager = AgentManager(base_data_dir=tmp_path)
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    lifecycle.restore_persisted_child(child_name, mandate)
    expiry_owner = lifecycle._tracked[child_name].ttl_task
    assert expiry_owner is not None
    assert registry.begin_retirement(child_name=child_name, child_did=child_did)

    assert lifecycle.retire_persisted_child(child_name) is True

    await asyncio.sleep(0)
    assert expiry_owner.cancelled()
    assert (child_name.casefold(), child_did) not in lifecycle._cold_ttl_tasks


@pytest.mark.asyncio
async def test_terminal_result_fences_descendant_spawns_before_tree_snapshot() -> None:
    """A terminal parent cannot snapshot past an admitted child spawn."""

    manager = _make_mock_manager()
    parent_did = "did:test:terminal-fence-parent"
    fence_token = object()
    fenced = False

    async def begin_fence(candidate_did):
        nonlocal fenced
        assert candidate_did == parent_did
        fenced = True
        return fence_token

    def get_children(candidate_did):
        assert candidate_did == parent_did
        assert fenced
        return []

    manager.begin_terminal_descendant_spawn_fence = AsyncMock(side_effect=begin_fence)
    manager.end_terminal_descendant_spawn_fence = MagicMock()
    manager.get_children = MagicMock(side_effect=get_children)
    lifecycle = SpawnedAgentLifecycle(manager)
    await lifecycle.register(
        "TerminalFenceParent",
        parent_did,
        "did:test:terminal-fence-root",
        ttl_seconds=0,
        mode=SpawnMode.PERSISTENT,
    )

    result = await lifecycle.report_result(
        "TerminalFenceParent",
        status=SpawnStatus.COMPLETED,
    )

    assert result is not None
    manager.begin_terminal_descendant_spawn_fence.assert_awaited_once_with(parent_did)
    manager.end_terminal_descendant_spawn_fence.assert_called_once_with(fence_token)


def test_expired_restored_ttl_handoff_raises_typed_retirement_signal() -> None:
    """The manager must recognize this boundary and write its tombstone."""

    manager = _make_mock_manager()
    lifecycle = SpawnedAgentLifecycle(manager)
    mandate = SpawnMandate(
        parent_did="did:test:expired-parent",
        child_did="did:test:expired-child",
        ttl_seconds=60,
    )
    lifecycle.restore_persisted_child(
        "ExpiredChild",
        mandate,
        arm_ttl=False,
    )
    lifecycle._remaining_ttl_seconds = MagicMock(return_value=0)

    with pytest.raises(
        PersistedSpawnMandateExpiredError,
        match="expired during onboarding",
    ):
        lifecycle.arm_restored_child_ttl(
            "ExpiredChild",
            expected_child_did=mandate.child_did,
        )

    lifecycle.withdraw_persisted_child("ExpiredChild")


@pytest.mark.asyncio
async def test_direct_retirement_is_not_owned_by_another_child_finalizer() -> None:
    """A's lifecycle operation must not strand B behind the global lock."""

    manager = _make_mock_manager()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def terminate_child(_parent_did, child_name, **_kwargs):
        assert child_name == "FinalizingA"
        entered.set()
        await release.wait()
        return True

    manager.terminate_child = AsyncMock(side_effect=terminate_child)
    lifecycle = SpawnedAgentLifecycle(manager)
    await lifecycle.register(
        "FinalizingA",
        "did:test:finalizing-a",
        "did:test:parent",
        ttl_seconds=0,
        mode=SpawnMode.PERSISTENT,
    )
    await lifecycle.register(
        "DirectB",
        "did:test:direct-b",
        "did:test:parent",
        ttl_seconds=0,
        mode=SpawnMode.PERSISTENT,
    )

    finalizer = asyncio.create_task(lifecycle.terminate("FinalizingA"))
    await entered.wait()
    assert lifecycle._lock.locked()

    assert lifecycle.retire_persisted_child(
        "DirectB", expected_child_did="did:test:direct-b"
    )
    assert not lifecycle.is_tracked("DirectB")
    assert lifecycle.is_tracked("FinalizingA")

    release.set()
    await finalizer
    assert lifecycle.get_tracked_children() == []


@pytest.mark.asyncio
async def test_manager_prune_cancels_removed_child_ttl_before_name_reuse() -> None:
    manager = AgentManager()
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    old_did = "did:test:removed-child"
    await lifecycle.register(
        "Reusable",
        old_did,
        "did:test:parent",
        ttl_seconds=3600,
    )
    old_task = lifecycle._tracked["Reusable"].ttl_task
    manager._parent_children["did:test:parent"] = ["Reusable"]
    manager._child_mandates["Reusable"] = SpawnMandate(
        parent_did="did:test:parent",
        child_did=old_did,
    )

    manager._prune_child_relationship_and_mandate(
        "did:test:parent",
        "Reusable",
    )
    assert not lifecycle.is_tracked("Reusable")
    await asyncio.sleep(0)
    await lifecycle.register(
        "Reusable",
        "did:test:replacement-child",
        "did:test:other-parent",
        ttl_seconds=3600,
    )

    assert old_task is not None and old_task.cancelled()
    assert lifecycle._tracked["Reusable"].child_did == "did:test:replacement-child"
    await lifecycle.shutdown()


@pytest.mark.asyncio
async def test_manager_prune_does_not_cancel_lifecycle_task_terminating_itself() -> None:
    manager = AgentManager()
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    child_did = "did:test:self-terminating-child"
    await lifecycle.register(
        "SelfTerminating",
        child_did,
        "did:test:parent",
        ttl_seconds=3600,
    )
    original_ttl = lifecycle._tracked["SelfTerminating"].ttl_task
    assert original_ttl is not None
    original_ttl.cancel()
    with pytest.raises(asyncio.CancelledError):
        await original_ttl
    lifecycle._tracked["SelfTerminating"].ttl_task = asyncio.current_task()
    manager._parent_children["did:test:parent"] = ["SelfTerminating"]
    manager._child_mandates["SelfTerminating"] = SpawnMandate(
        parent_did="did:test:parent",
        child_did=child_did,
    )

    manager._prune_child_relationship_and_mandate(
        "did:test:parent",
        "SelfTerminating",
    )

    assert not lifecycle.is_tracked("SelfTerminating")
    assert not asyncio.current_task().cancelling()


@pytest.mark.asyncio
async def test_direct_prune_does_not_cancel_same_child_finalizer_waiting_for_lock() -> None:
    """A queued TTL owner finishes reconciliation after direct removal."""

    manager = AgentManager()
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    entered = asyncio.Event()
    release = asyncio.Event()

    async def terminate_child(_parent_did, child_name, **_kwargs):
        if child_name == "BlockingA":
            entered.set()
            await release.wait()
        return True

    manager.terminate_child = AsyncMock(side_effect=terminate_child)
    await lifecycle.register(
        "BlockingA",
        "did:test:blocking-a",
        "did:test:parent",
        ttl_seconds=0,
        mode=SpawnMode.PERSISTENT,
    )
    await lifecycle.register(
        "QueuedB",
        "did:test:queued-b",
        "did:test:parent",
        ttl_seconds=3600,
    )
    original_timer = lifecycle._tracked["QueuedB"].ttl_task
    assert original_timer is not None
    original_timer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await original_timer

    blocker = asyncio.create_task(lifecycle.terminate("BlockingA"))
    await entered.wait()
    queued = asyncio.create_task(
        lifecycle._ttl_monitor("QueuedB", "did:test:queued-b", 0)
    )
    lifecycle._tracked["QueuedB"].ttl_task = queued
    for _ in range(20):
        if lifecycle._finalization_owner_counts.get(
            ("QueuedB", "did:test:queued-b")
        ):
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("queued TTL finalizer never claimed ownership")

    manager._parent_children["did:test:parent"] = ["QueuedB"]
    manager._child_mandates["QueuedB"] = SpawnMandate(
        parent_did="did:test:parent",
        child_did="did:test:queued-b",
    )
    manager._prune_child_relationship_and_mandate(
        "did:test:parent",
        "QueuedB",
    )

    release.set()
    await blocker
    await queued
    assert not lifecycle.is_tracked("QueuedB")
    assert ("QueuedB", "did:test:queued-b") not in lifecycle._finalization_owner_counts


@pytest.mark.asyncio
async def test_cancelled_queued_finalizer_retires_directly_pruned_child() -> None:
    manager = AgentManager()
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    lock_held = asyncio.Event()
    release_lock = asyncio.Event()

    async def hold_lifecycle_lock():
        async with lifecycle._lock:
            lock_held.set()
            await release_lock.wait()

    await lifecycle.register(
        "Queued",
        "did:test:queued",
        "did:test:parent",
        ttl_seconds=3600,
    )
    original_timer = lifecycle._tracked["Queued"].ttl_task
    assert original_timer is not None
    original_timer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await original_timer

    blocker = asyncio.create_task(hold_lifecycle_lock())
    await lock_held.wait()

    queued = asyncio.create_task(
        lifecycle._ttl_monitor("Queued", "did:test:queued", 0)
    )
    lifecycle._tracked["Queued"].ttl_task = queued
    for _ in range(20):
        if lifecycle._finalization_owner_counts.get(
            ("Queued", "did:test:queued")
        ):
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("queued finalizer never claimed ownership")

    manager._parent_children["did:test:parent"] = ["Queued"]
    manager._child_mandates["Queued"] = SpawnMandate(
        parent_did="did:test:parent",
        child_did="did:test:queued",
    )
    manager._prune_child_relationship_and_mandate(
        "did:test:parent",
        "Queued",
    )

    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued

    assert not lifecycle.is_tracked("Queued")
    assert ("Queued", "did:test:queued") not in lifecycle._finalization_owner_counts
    release_lock.set()
    await blocker


@pytest.mark.asyncio
async def test_stale_ttl_monitor_cannot_terminate_same_name_replacement() -> None:
    manager = _make_mock_manager()
    lifecycle = SpawnedAgentLifecycle(manager)
    await lifecycle.register(
        "Reusable",
        "did:test:replacement-child",
        "did:test:parent",
        ttl_seconds=3600,
    )

    await lifecycle._ttl_monitor(
        "Reusable",
        "did:test:removed-child",
        0,
    )

    manager.terminate_child.assert_not_awaited()
    assert lifecycle._tracked["Reusable"].child_did == "did:test:replacement-child"
    await lifecycle.shutdown()


class TestSpawnResult:
    """SpawnResult dataclass basics."""

    def test_defaults(self):
        result = SpawnResult(
            child_name="worker",
            child_did="did:child",
            status=SpawnStatus.COMPLETED,
        )
        assert result.child_name == "worker"
        assert result.status == SpawnStatus.COMPLETED
        assert result.output_artifacts == {}
        assert result.budget_consumed == Decimal("0")
        assert result.ended_at  # non-empty
        assert result.finalized_from_absence is False

    def test_with_artifacts(self):
        result = SpawnResult(
            child_name="worker",
            child_did="did:child",
            status=SpawnStatus.COMPLETED,
            output_artifacts={"summary": "done", "files": ["out.txt"]},
            budget_consumed=Decimal("3.50"),
        )
        assert result.output_artifacts["summary"] == "done"
        assert result.budget_consumed == Decimal("3.50")


class TestSpawnStatus:
    """SpawnStatus enum values."""

    def test_all_statuses(self):
        assert SpawnStatus.RUNNING == "running"
        assert SpawnStatus.COMPLETED == "completed"
        assert SpawnStatus.TERMINATED == "terminated"
        assert SpawnStatus.TIMED_OUT == "timed_out"
        assert SpawnStatus.FAILED == "failed"


class TestSpawnMode:
    """SpawnMode enum values."""

    def test_modes(self):
        assert SpawnMode.EPHEMERAL == "ephemeral"
        assert SpawnMode.PERSISTENT == "persistent"


class TestLifecycleRegistration:
    """Test child registration and tracking."""

    @pytest.mark.asyncio
    async def test_register_tracks_child(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
            purpose="test task",
        )

        assert lifecycle.is_tracked("worker")
        assert "worker" in lifecycle.get_tracked_children()

        # Clean up
        await lifecycle.shutdown()

    @pytest.mark.asyncio
    async def test_register_starts_ttl_task(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        tracked = lifecycle._tracked["worker"]
        assert tracked.ttl_task is not None
        assert not tracked.ttl_task.done()

        await lifecycle.shutdown()


class TestTTLExpiration:
    """TTL expiration triggers auto-termination."""

    @pytest.mark.asyncio
    async def test_ttl_expiry_auto_terminates(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        # Use a very short TTL
        await lifecycle.register(
            child_name="ephemeral",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=0.1,  # 100ms
        )

        # Wait for TTL to expire
        await asyncio.sleep(0.3)

        # Child should have been terminated
        assert not lifecycle.is_tracked("ephemeral")

        # Result should be stored with TIMED_OUT status
        result = lifecycle.get_result("ephemeral")
        assert result is not None
        assert result.status == SpawnStatus.TIMED_OUT
        assert result.child_name == "ephemeral"

        # AgentManager.terminate_child should have been called
        manager.terminate_child.assert_awaited_once_with("did:parent", "ephemeral")

    @pytest.mark.asyncio
    async def test_ttl_expiry_records_auto_discovery_retirement(self):
        class RetirementRecordingManager:
            def __init__(self):
                self._child_mandates = {}
                self._parent_children = {}
                self.retirements = []

            async def terminate_child(self, _parent_did, _child_name):
                return True

            def get_agent(self, _child_name):
                return None

            def get_children(self, _parent_did):
                return []

            def record_expired_spawn_retirement(
                self, child_name, *, expected_child_did
            ):
                self.retirements.append((child_name, expected_child_did))

        manager = RetirementRecordingManager()
        lifecycle = SpawnedAgentLifecycle(manager)
        await lifecycle.register(
            child_name="retired-after-ttl",
            child_did="did:test:retired-after-ttl",
            parent_did="did:test:parent",
            ttl_seconds=0.01,
        )

        for _ in range(100):
            if not lifecycle.is_tracked("retired-after-ttl"):
                break
            await asyncio.sleep(0.01)

        assert manager.retirements == [
            ("retired-after-ttl", "did:test:retired-after-ttl")
        ]

    @pytest.mark.asyncio
    async def test_ttl_cancelled_on_early_completion(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="quick",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=100,  # Long TTL
        )

        ttl_task = lifecycle._tracked["quick"].ttl_task

        # Report result before TTL
        await lifecycle.report_result(
            child_name="quick",
            output_artifacts={"answer": 42},
            status=SpawnStatus.COMPLETED,
        )

        # Let the cancellation propagate
        await asyncio.sleep(0.05)

        # TTL task should be cancelled or done
        assert ttl_task.done()

        # Child should be cleaned up
        assert not lifecycle.is_tracked("quick")

        result = lifecycle.get_result("quick")
        assert result.status == SpawnStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_ttl_reaper_reconciles_grouped_retained_offboarding(self):
        manager = _make_mock_manager()
        retained = RuntimeOffboardingRetainedError(
            agent_name="ephemeral",
            agent_id="did:child",
            runtime_path=Path("operator/runtime/child"),
            cause=OSError("retained"),
        )
        manager.terminate_child.side_effect = BaseExceptionGroup(
            "cancelled retained cleanup",
            [asyncio.CancelledError(), retained],
        )
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="ephemeral",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=0.01,
        )
        ttl_task = lifecycle._tracked["ephemeral"].ttl_task
        await asyncio.sleep(0.1)

        assert ttl_task.done()
        assert ttl_task.exception() is None
        assert not lifecycle.is_tracked("ephemeral")
        assert lifecycle.get_result("ephemeral").status == SpawnStatus.TIMED_OUT

    @pytest.mark.asyncio
    async def test_refused_ttl_termination_rearms_and_publishes_only_on_retry(self):
        manager = _make_mock_manager()
        manager.get_agent.return_value = object()
        manager.get_children.return_value = ["retry-ttl"]
        first_refused = asyncio.Event()
        allow_retry = asyncio.Event()
        attempts = 0

        async def terminate_child(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                first_refused.set()
                return False
            await allow_retry.wait()
            return True

        manager.terminate_child.side_effect = terminate_child
        lifecycle = SpawnedAgentLifecycle(manager)
        await lifecycle.register(
            child_name="retry-ttl",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=0.01,
        )
        first_ttl = lifecycle._tracked["retry-ttl"].ttl_task

        await asyncio.wait_for(first_refused.wait(), timeout=1)
        await asyncio.sleep(0)

        assert lifecycle.is_tracked("retry-ttl")
        assert lifecycle.get_result("retry-ttl") is None
        retry_ttl = lifecycle._tracked["retry-ttl"].ttl_task
        assert retry_ttl is not first_ttl
        assert retry_ttl is not None and not retry_ttl.done()

        allow_retry.set()
        for _ in range(100):
            if not lifecycle.is_tracked("retry-ttl"):
                break
            await asyncio.sleep(0.01)

        assert not lifecycle.is_tracked("retry-ttl")
        assert manager.terminate_child.await_count == 2
        assert lifecycle.get_result("retry-ttl").status is SpawnStatus.TIMED_OUT

    @pytest.mark.asyncio
    async def test_ttl_finalizes_child_already_removed_by_real_manager(self, tmp_path):
        """A pruned parent edge is already-gone evidence, not a refusal."""

        manager = AgentManager()
        child = MagicMock()
        child.agent_id = "did:child:already-gone"
        child.shutdown = AsyncMock()
        manager._agents["already-gone"] = child
        manager._agent_names[child.agent_id] = "already-gone"
        manager._parent_children["did:parent"] = ["already-gone"]
        lifecycle = SpawnedAgentLifecycle(manager)
        ephemeral_dir = tmp_path / "already-gone"
        ephemeral_dir.mkdir()
        (ephemeral_dir / "artifact").write_text("temporary")

        await lifecycle.register(
            child_name="already-gone",
            child_did=child.agent_id,
            parent_did="did:parent",
            ttl_seconds=0.05,
            mode=SpawnMode.EPHEMERAL,
            temp_dir=str(ephemeral_dir),
        )
        lifecycle._fire_hook = AsyncMock()
        assert await manager.remove_agent("already-gone") is True
        assert manager.get_agent("already-gone") is None
        assert manager.get_children("did:parent") == []

        for _ in range(100):
            if not lifecycle.is_tracked("already-gone"):
                break
            await asyncio.sleep(0.01)

        assert not lifecycle.is_tracked("already-gone")
        assert lifecycle.get_result("already-gone").status is SpawnStatus.TIMED_OUT
        assert not ephemeral_dir.exists()
        lifecycle._fire_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ttl_refusal_retries_are_bounded_and_explicitly_retryable(
        self, tmp_path, caplog
    ):
        """A live refused child stops auto-looping but remains operator-retryable."""

        manager = _make_mock_manager()
        manager.get_agent.return_value = object()
        manager.get_children.return_value = ["bounded-refusal"]
        manager.terminate_child.return_value = False
        lifecycle = SpawnedAgentLifecycle(manager)
        ephemeral_dir = tmp_path / "bounded-refusal"
        ephemeral_dir.mkdir()

        await lifecycle.register(
            child_name="bounded-refusal",
            child_did="did:child:bounded-refusal",
            parent_did="did:parent",
            ttl_seconds=0.01,
            mode=SpawnMode.EPHEMERAL,
            temp_dir=str(ephemeral_dir),
        )
        lifecycle._fire_hook = AsyncMock()
        for _ in range(100):
            refusal = lifecycle.get_termination_refusal("bounded-refusal")
            if refusal is not None:
                break
            await asyncio.sleep(0.01)

        refusal = lifecycle.get_termination_refusal("bounded-refusal")
        assert refusal is not None
        recorded_at = refusal.pop("recorded_at")
        assert recorded_at
        assert refusal == {
            "termination_not_performed": True,
            "automatic_termination_attempts": 3,
            "automatic_retries_exhausted": True,
            "operator_action_required": True,
            "retry_termination": True,
            "requested_status": "timed_out",
        }
        assert lifecycle.get_result("bounded-refusal") is None
        assert lifecycle._tracked["bounded-refusal"].result is None
        assert manager.terminate_child.await_count == 3
        await asyncio.sleep(0.05)
        assert manager.terminate_child.await_count == 3
        assert lifecycle.is_tracked("bounded-refusal")
        assert ephemeral_dir.exists()
        lifecycle._fire_hook.assert_not_awaited()
        assert sum(
            "periodic retry remain active" in record.getMessage()
            for record in caplog.records
        ) == 2
        assert sum(
            "automatic retries stopped" in record.getMessage()
            for record in caplog.records
        ) == 1

        manager.terminate_child.return_value = True
        retried = await lifecycle.terminate("bounded-refusal")

        assert retried is not None
        assert retried.status is SpawnStatus.TERMINATED
        assert manager.terminate_child.await_count == 4
        assert not lifecycle.is_tracked("bounded-refusal")
        assert lifecycle.get_termination_refusal("bounded-refusal") is None
        assert not ephemeral_dir.exists()
        lifecycle._fire_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_real_result_supersedes_bounded_ttl_refusal(self, tmp_path):
        """Operator refusal state cannot consume child artifacts or budget."""

        manager = _make_mock_manager()
        manager.get_agent.return_value = object()
        manager.get_children.return_value = ["completed-after-refusal"]
        manager.terminate_child.return_value = False
        lifecycle = SpawnedAgentLifecycle(manager)
        ephemeral_dir = tmp_path / "completed-after-refusal"
        ephemeral_dir.mkdir()
        await lifecycle.register(
            child_name="completed-after-refusal",
            child_did="did:child:completed-after-refusal",
            parent_did="did:parent",
            ttl_seconds=0.01,
            mode=SpawnMode.EPHEMERAL,
            temp_dir=str(ephemeral_dir),
        )
        lifecycle._fire_hook = AsyncMock()
        for _ in range(100):
            if lifecycle.get_termination_refusal("completed-after-refusal"):
                break
            await asyncio.sleep(0.01)

        assert lifecycle.get_termination_refusal("completed-after-refusal")
        assert lifecycle.get_result("completed-after-refusal") is None
        manager.terminate_child.return_value = True

        result = await lifecycle.report_result(
            "completed-after-refusal",
            output_artifacts={"answer": "42"},
            budget_consumed=Decimal("1.75"),
        )

        assert result is not None
        assert result.status is SpawnStatus.COMPLETED
        assert result.output_artifacts == {"answer": "42"}
        assert result.budget_consumed == Decimal("1.75")
        assert lifecycle.get_result("completed-after-refusal") is result
        assert lifecycle.get_termination_refusal("completed-after-refusal") is None
        assert not lifecycle.is_tracked("completed-after-refusal")
        assert not ephemeral_dir.exists()
        lifecycle._fire_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refused_result_report_preserves_operator_state(self, tmp_path):
        """A work result cannot erase a still-live child's refusal witness."""

        manager = _make_mock_manager()
        manager.get_agent.return_value = object()
        manager.get_children.return_value = ["completed-but-live"]
        manager.terminate_child.return_value = False
        lifecycle = SpawnedAgentLifecycle(manager)
        ephemeral_dir = tmp_path / "completed-but-live"
        ephemeral_dir.mkdir()
        await lifecycle.register(
            child_name="completed-but-live",
            child_did="did:child:completed-but-live",
            parent_did="did:parent",
            ttl_seconds=0.01,
            mode=SpawnMode.EPHEMERAL,
            temp_dir=str(ephemeral_dir),
        )
        lifecycle._fire_hook = AsyncMock()
        for _ in range(100):
            refusal = lifecycle.get_termination_refusal("completed-but-live")
            if refusal is not None:
                break
            await asyncio.sleep(0.01)

        refusal = lifecycle.get_termination_refusal("completed-but-live")
        assert refusal is not None
        assert refusal["automatic_retries_exhausted"] is True

        result = await lifecycle.report_result(
            "completed-but-live",
            output_artifacts={"answer": "not-finalized"},
            budget_consumed=Decimal("2.25"),
        )

        assert result is None
        assert lifecycle.get_termination_refusal("completed-but-live") == refusal
        assert lifecycle.get_result("completed-but-live") is None
        assert lifecycle.is_tracked("completed-but-live")
        assert ephemeral_dir.exists()
        lifecycle._fire_hook.assert_not_awaited()

        manager.terminate_child.return_value = True
        finalized = await lifecycle.terminate("completed-but-live")
        assert finalized is not None
        assert lifecycle.get_termination_refusal("completed-but-live") is None


class TestResultCollection:
    """Result reporting from child to parent."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [SpawnStatus.COMPLETED, SpawnStatus.FAILED])
    async def test_terminal_result_records_durable_restart_retirement(self, status):
        manager = _make_mock_manager()
        manager.record_expired_spawn_retirement = MagicMock()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="terminal-worker",
            child_did="did:child:terminal-worker",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        result = await lifecycle.report_result(
            child_name="terminal-worker",
            status=status,
        )

        assert result is not None
        manager.record_expired_spawn_retirement.assert_called_once_with(
            "terminal-worker",
            expected_child_did="did:child:terminal-worker",
        )

    @pytest.mark.asyncio
    async def test_terminal_parent_retires_descendant_restart_authority(self, tmp_path):
        """A completed parent cannot leave a grandchild restart witness live."""

        root_did = "did:test:terminal-root"
        parent_name = "TerminalParent"
        parent_did = "did:test:terminal-parent"
        descendant_name = "TerminalDescendant"
        descendant_did = "did:test:terminal-descendant"
        parent_mandate = SpawnMandate(
            parent_did=root_did,
            child_did=parent_did,
            ttl_seconds=0,
            parent_signature="signed-parent-witness",
        )
        descendant_mandate = SpawnMandate(
            parent_did=parent_did,
            child_did=descendant_did,
            ttl_seconds=0,
            parent_signature="signed-descendant-witness",
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        for name, did, mandate, port in (
            (parent_name, parent_did, parent_mandate, 8802),
            (descendant_name, descendant_did, descendant_mandate, 8803),
        ):
            registry.record_active(
                child_name=name,
                child_did=did,
                mandate=mandate,
                config=LocalAgentConfig(
                    data_dir=Path("agent_data") / name,
                    port=port,
                ),
            )

        manager = AgentManager(base_data_dir=tmp_path)
        parent = MagicMock(agent_id=parent_did)
        parent.agent_id = parent_did
        parent.shutdown = AsyncMock()
        descendant = MagicMock(agent_id=descendant_did)
        descendant.agent_id = descendant_did
        descendant.shutdown = AsyncMock()
        manager._agents.update({parent_name: parent, descendant_name: descendant})
        manager._agent_names.update(
            {parent_did: parent_name, descendant_did: descendant_name}
        )
        manager._parent_children.update(
            {root_did: [parent_name], parent_did: [descendant_name]}
        )
        manager._child_mandates.update(
            {
                parent_name: parent_mandate,
                descendant_name: descendant_mandate,
            }
        )
        lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle = lifecycle
        await lifecycle.register(
            child_name=parent_name,
            child_did=parent_did,
            parent_did=root_did,
            ttl_seconds=0,
            mode=SpawnMode.PERSISTENT,
        )

        result = await lifecycle.report_result(
            parent_name,
            status=SpawnStatus.COMPLETED,
        )

        assert result is not None
        assert manager.get_agent(parent_name) is None
        assert manager.get_agent(descendant_name) is None
        assert registry.get(parent_did).state == "retired"
        assert registry.get(descendant_did).state == "retired"
        restarted = AgentManager(base_data_dir=tmp_path)
        assert (
            restarted._reconcile_spawn_authority_restart_roster(
                MultiAgentConfig(agents={})
            ).agents
            == {}
        )

    @pytest.mark.asyncio
    async def test_terminal_parent_retires_previously_stopped_descendant(self, tmp_path):
        """A retained cold child remains inside its parent's terminal tree."""

        root_did = "did:test:stopped-tree-root"
        parent_name = "StoppedTreeParent"
        parent_did = "did:test:stopped-tree-parent"
        descendant_name = "StoppedTreeDescendant"
        descendant_did = "did:test:stopped-tree-descendant"
        parent_mandate = SpawnMandate(
            parent_did=root_did,
            child_did=parent_did,
            ttl_seconds=0,
            parent_signature="signed-stopped-tree-parent",
        )
        descendant_mandate = SpawnMandate(
            parent_did=parent_did,
            child_did=descendant_did,
            ttl_seconds=0,
            parent_signature="signed-stopped-tree-descendant",
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        for name, did, mandate, port in (
            (parent_name, parent_did, parent_mandate, 8802),
            (descendant_name, descendant_did, descendant_mandate, 8803),
        ):
            registry.record_active(
                child_name=name,
                child_did=did,
                mandate=mandate,
                config=LocalAgentConfig(
                    data_dir=Path("agent_data") / name,
                    port=port,
                ),
            )

        manager = AgentManager(base_data_dir=tmp_path)
        parent = MagicMock(agent_id=parent_did)
        parent.agent_id = parent_did
        parent.shutdown = AsyncMock()
        descendant = MagicMock(agent_id=descendant_did)
        descendant.agent_id = descendant_did
        descendant.shutdown = AsyncMock()
        manager._agents.update({parent_name: parent, descendant_name: descendant})
        manager._agent_names.update(
            {parent_did: parent_name, descendant_did: descendant_name}
        )
        manager._parent_children.update(
            {root_did: [parent_name], parent_did: [descendant_name]}
        )
        manager._child_mandates.update(
            {
                parent_name: parent_mandate,
                descendant_name: descendant_mandate,
            }
        )
        lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle = lifecycle
        await lifecycle.register(
            child_name=parent_name,
            child_did=parent_did,
            parent_did=root_did,
            ttl_seconds=0,
            mode=SpawnMode.PERSISTENT,
        )

        assert await manager.terminate_child(parent_did, descendant_name) is True
        assert manager.get_children(parent_did) == []
        assert registry.get(descendant_did).active

        result = await lifecycle.report_result(
            parent_name,
            status=SpawnStatus.COMPLETED,
        )

        assert result is not None
        assert registry.get(parent_did).retired
        assert registry.get(descendant_did).retired

    @pytest.mark.asyncio
    async def test_terminal_tree_retirement_intent_is_one_durable_write(self, tmp_path):
        """A crash after intent persistence cannot leave a descendant active."""

        root_did = "did:test:atomic-terminal-root"
        parent_name = "AtomicTerminalParent"
        parent_did = "did:test:atomic-terminal-parent"
        descendant_name = "AtomicTerminalDescendant"
        descendant_did = "did:test:atomic-terminal-descendant"
        parent_mandate = SpawnMandate(
            parent_did=root_did,
            child_did=parent_did,
            ttl_seconds=0,
            parent_signature="signed-atomic-parent",
        )
        descendant_mandate = SpawnMandate(
            parent_did=parent_did,
            child_did=descendant_did,
            ttl_seconds=0,
            parent_signature="signed-atomic-descendant",
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        for name, did, mandate, port in (
            (parent_name, parent_did, parent_mandate, 8802),
            (descendant_name, descendant_did, descendant_mandate, 8803),
        ):
            registry.record_active(
                child_name=name,
                child_did=did,
                mandate=mandate,
                config=LocalAgentConfig(
                    data_dir=Path("agent_data") / name,
                    port=port,
                ),
            )

        manager = AgentManager(base_data_dir=tmp_path)
        parent = MagicMock(agent_id=parent_did)
        parent.agent_id = parent_did
        descendant = MagicMock(agent_id=descendant_did)
        descendant.agent_id = descendant_did
        manager._agents.update({parent_name: parent, descendant_name: descendant})
        manager._agent_names.update(
            {parent_did: parent_name, descendant_did: descendant_name}
        )
        manager._parent_children.update(
            {root_did: [parent_name], parent_did: [descendant_name]}
        )
        manager._child_mandates.update(
            {
                parent_name: parent_mandate,
                descendant_name: descendant_mandate,
            }
        )
        lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle = lifecycle
        await lifecycle.register(
            child_name=parent_name,
            child_did=parent_did,
            parent_did=root_did,
            ttl_seconds=0,
            mode=SpawnMode.PERSISTENT,
        )
        authority_registry = manager._spawn_authority_registry
        save = authority_registry._save

        def save_then_crash(records, pending):
            save(records, pending)
            raise SystemExit("simulated crash after retirement intent write")

        with patch.object(
            authority_registry,
            "_save",
            side_effect=save_then_crash,
        ), pytest.raises(SystemExit, match="simulated crash"):
            await lifecycle.report_result(
                parent_name,
                status=SpawnStatus.COMPLETED,
            )

        durable = SpawnAuthorityRegistry(tmp_path)
        assert durable.get(parent_did).state == "retiring"
        assert durable.get(descendant_did).state == "retiring"

    @pytest.mark.asyncio
    async def test_crash_after_manager_removal_leaves_restart_denied(self, tmp_path):
        """A crash before the final tombstone cannot resurrect a result child."""

        child_name = "crash-window"
        child_did = "did:test:crash-window-child"
        parent_did = "did:test:crash-window-parent"
        private_key, _ = generate_secp256k1_keypair()
        mandate = sign_mandate(
            SpawnMandate(
                parent_did=parent_did,
                child_did=child_did,
                ttl_seconds=0,
            ),
            private_key,
        )
        config = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=config,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        child = MagicMock(agent_id=child_did)
        child.agent_id = child_did
        manager._agents[child_name] = child
        manager._agent_names[child_did] = child_name
        manager._parent_children[parent_did] = [child_name]
        manager._child_mandates[child_name] = mandate

        async def remove_then_return(_parent_did, _child_name):
            manager._agents.pop(child_name)
            manager._agent_names.pop(child_did)
            manager._parent_children[parent_did].remove(child_name)
            manager._child_mandates.pop(child_name)
            return True

        manager.terminate_child = AsyncMock(side_effect=remove_then_return)
        manager.record_expired_spawn_retirement = MagicMock(
            side_effect=SystemExit("simulated process crash")
        )
        lifecycle = SpawnedAgentLifecycle(manager)
        await lifecycle.register(
            child_name=child_name,
            child_did=child_did,
            parent_did=parent_did,
            ttl_seconds=0,
            mode=SpawnMode.PERSISTENT,
        )

        with pytest.raises(SystemExit, match="simulated process crash"):
            await lifecycle.report_result(child_name, status=SpawnStatus.COMPLETED)

        witness = registry.get(child_did)
        assert witness is not None and witness.state == "retiring"
        restarted = AgentManager(base_data_dir=tmp_path)
        assert (
            restarted._reconcile_spawn_authority_restart_roster(
                MultiAgentConfig(agents={})
            ).agents
            == {}
        )

    @pytest.mark.asyncio
    async def test_live_terminal_refusal_reopens_restart_authority(self, tmp_path):
        child_name = "refused-window"
        child_did = "did:test:refused-window-child"
        parent_did = "did:test:refused-window-parent"
        private_key, _ = generate_secp256k1_keypair()
        mandate = sign_mandate(
            SpawnMandate(
                parent_did=parent_did,
                child_did=child_did,
                ttl_seconds=0,
            ),
            private_key,
        )
        config = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=config,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        child = MagicMock(agent_id=child_did)
        child.agent_id = child_did
        manager._agents[child_name] = child
        manager._agent_names[child_did] = child_name
        manager._parent_children[parent_did] = [child_name]
        manager._child_mandates[child_name] = mandate
        manager.terminate_child = AsyncMock(return_value=False)
        lifecycle = SpawnedAgentLifecycle(manager)
        await lifecycle.register(
            child_name=child_name,
            child_did=child_did,
            parent_did=parent_did,
            ttl_seconds=0,
            mode=SpawnMode.PERSISTENT,
        )

        assert await lifecycle.report_result(
            child_name,
            status=SpawnStatus.FAILED,
        ) is None
        witness = registry.get(child_did)
        assert witness is not None and witness.active

    @pytest.mark.asyncio
    async def test_report_result_stores_result(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        result = await lifecycle.report_result(
            child_name="worker",
            output_artifacts={"data": [1, 2, 3]},
            budget_consumed=Decimal("1.25"),
            status=SpawnStatus.COMPLETED,
        )

        assert result is not None
        assert result.status == SpawnStatus.COMPLETED
        assert result.output_artifacts == {"data": [1, 2, 3]}
        assert result.budget_consumed == Decimal("1.25")

    @pytest.mark.asyncio
    async def test_report_result_untracked_returns_none(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        result = await lifecycle.report_result(
            child_name="ghost",
            output_artifacts={},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pop_result_removes_it(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        await lifecycle.report_result(child_name="worker")

        result = lifecycle.pop_result("worker")
        assert result is not None
        assert lifecycle.get_result("worker") is None

    @pytest.mark.asyncio
    async def test_report_failed_status(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="broken",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        result = await lifecycle.report_result(
            child_name="broken",
            status=SpawnStatus.FAILED,
        )

        assert result.status == SpawnStatus.FAILED


class TestEphemeralCleanup:
    """Ephemeral cleanup — no leftover temp files."""

    @pytest.mark.asyncio
    async def test_ephemeral_cleanup_removes_temp_dir(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        # Create a real temp directory
        temp_dir = tempfile.mkdtemp(prefix="kestrel_test_")
        # Put a file in it to verify cleanup
        with open(os.path.join(temp_dir, "test.txt"), "w") as f:
            f.write("ephemeral data")

        assert os.path.exists(temp_dir)

        await lifecycle.register(
            child_name="temp_worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
            mode=SpawnMode.EPHEMERAL,
            temp_dir=temp_dir,
        )

        # Terminate triggers cleanup
        await lifecycle.terminate("temp_worker", reason="test cleanup")

        # Temp dir should be gone
        assert not os.path.exists(temp_dir)

    @pytest.mark.asyncio
    async def test_persistent_mode_keeps_data(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        temp_dir = tempfile.mkdtemp(prefix="kestrel_test_persist_")

        await lifecycle.register(
            child_name="persist_worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
            mode=SpawnMode.PERSISTENT,
            temp_dir=temp_dir,
        )

        await lifecycle.terminate("persist_worker")

        # Persistent mode should NOT clean up the directory
        assert os.path.exists(temp_dir)

        # Manual cleanup
        os.rmdir(temp_dir)

    @pytest.mark.asyncio
    async def test_ephemeral_ttl_cleanup(self):
        """TTL expiry also cleans up ephemeral resources."""
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        temp_dir = tempfile.mkdtemp(prefix="kestrel_test_ttl_")

        await lifecycle.register(
            child_name="ttl_worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=0.1,
            mode=SpawnMode.EPHEMERAL,
            temp_dir=temp_dir,
        )

        await asyncio.sleep(0.3)

        assert not os.path.exists(temp_dir)

    @pytest.mark.asyncio
    async def test_create_ephemeral_dir(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        temp_dir = lifecycle.create_ephemeral_dir()
        assert os.path.isdir(temp_dir)
        assert "kestrel_spawn_" in temp_dir

        # Cleanup
        os.rmdir(temp_dir)


class TestExplicitTermination:
    """Explicit terminate() method."""

    @pytest.mark.asyncio
    async def test_terminate_returns_result(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="doomed",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        result = await lifecycle.terminate("doomed", reason="no longer needed")

        assert result is not None
        assert result.status == SpawnStatus.TERMINATED
        assert not lifecycle.is_tracked("doomed")

    @pytest.mark.asyncio
    async def test_restartable_stop_keeps_original_expiry_owner(
        self, monkeypatch, tmp_path
    ):
        """Stopping work cannot turn a finite mandate into a permanent hold."""

        child_name = "stopped-until-expiry"
        child_did = "did:test:stopped-until-expiry"
        parent_did = "did:test:stopped-parent"
        mandate = SpawnMandate(
            parent_did=parent_did,
            child_did=child_did,
            ttl_seconds=1,
            parent_signature="signed-stopped-child",
        )
        config = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=config,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        child = MagicMock()
        child.did = child_did
        child.agent_id = child_did
        child.shutdown = AsyncMock()
        manager._agents[child_name] = child
        manager._agent_names[child_did] = child_name
        manager._parent_children[parent_did] = [child_name]
        manager._child_mandates[child_name] = mandate
        manager._created_configs[child_name] = config
        lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle = lifecycle
        monkeypatch.setattr(
            lifecycle,
            "_remaining_ttl_seconds",
            lambda _created_at, _ttl_seconds: 0.05,
        )
        await lifecycle.register(
            child_name=child_name,
            child_did=child_did,
            parent_did=parent_did,
            ttl_seconds=mandate.ttl_seconds,
            mode=SpawnMode.EPHEMERAL,
            started_at=mandate.created_at,
        )
        expiry_owner = lifecycle._tracked[child_name].ttl_task

        result = await lifecycle.terminate(child_name)

        assert result is not None
        assert not lifecycle.is_tracked(child_name)
        assert expiry_owner is not None
        assert not expiry_owner.cancelled()
        await asyncio.wait_for(asyncio.shield(expiry_owner), timeout=0.5)
        assert registry.get(child_did).retired
        assert manager._spawn_cap_slots_in_use() == 0

    @pytest.mark.asyncio
    async def test_stopped_parent_expiry_retires_cold_descendant_authority(
        self, monkeypatch, tmp_path
    ):
        """A stopped parent's eventual expiry closes its durable subtree."""

        root_did = "did:test:stopped-expiry-root"
        parent_name = "StoppedExpiryParent"
        parent_did = "did:test:stopped-expiry-parent"
        descendant_name = "StoppedExpiryDescendant"
        descendant_did = "did:test:stopped-expiry-descendant"
        parent_mandate = SpawnMandate(
            parent_did=root_did,
            child_did=parent_did,
            ttl_seconds=1,
            parent_signature="signed-stopped-expiry-parent",
        )
        descendant_mandate = SpawnMandate(
            parent_did=parent_did,
            child_did=descendant_did,
            ttl_seconds=3600,
            parent_signature="signed-stopped-expiry-descendant",
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        for name, did, mandate, port in (
            (parent_name, parent_did, parent_mandate, 8802),
            (descendant_name, descendant_did, descendant_mandate, 8803),
        ):
            registry.record_active(
                child_name=name,
                child_did=did,
                mandate=mandate,
                config=LocalAgentConfig(
                    data_dir=Path("agent_data") / name,
                    port=port,
                ),
            )

        manager = AgentManager(base_data_dir=tmp_path)
        parent = MagicMock(agent_id=parent_did)
        parent.agent_id = parent_did
        parent.shutdown = AsyncMock()
        descendant = MagicMock(agent_id=descendant_did)
        descendant.agent_id = descendant_did
        descendant.shutdown = AsyncMock()
        manager._agents.update({parent_name: parent, descendant_name: descendant})
        manager._agent_names.update(
            {parent_did: parent_name, descendant_did: descendant_name}
        )
        manager._parent_children.update(
            {root_did: [parent_name], parent_did: [descendant_name]}
        )
        manager._child_mandates.update(
            {
                parent_name: parent_mandate,
                descendant_name: descendant_mandate,
            }
        )
        lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle = lifecycle
        monkeypatch.setattr(
            lifecycle,
            "_remaining_ttl_seconds",
            lambda _created_at, _ttl_seconds: 0.01,
        )
        await lifecycle.register(
            child_name=parent_name,
            child_did=parent_did,
            parent_did=root_did,
            ttl_seconds=parent_mandate.ttl_seconds,
            mode=SpawnMode.EPHEMERAL,
            started_at=parent_mandate.created_at,
        )
        expiry_owner = lifecycle._tracked[parent_name].ttl_task

        result = await lifecycle.terminate(parent_name)

        assert result is not None
        assert expiry_owner is not None
        await asyncio.wait_for(asyncio.shield(expiry_owner), timeout=0.5)
        assert registry.get(parent_did).retired
        assert registry.get(descendant_did).retired
        assert manager._spawn_cap_slots_in_use() == 0

    @pytest.mark.asyncio
    async def test_destructive_stop_cancels_expiry_owner(self):
        """Destructive offboarding has no witness for the timer to retire."""

        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)
        await lifecycle.register(
            child_name="offboarded",
            child_did="did:offboarded",
            parent_did="did:parent",
            ttl_seconds=3600,
        )
        expiry_owner = lifecycle._tracked["offboarded"].ttl_task

        result = await lifecycle.terminate("offboarded", offboard_runtime=True)

        assert result is not None
        assert expiry_owner is not None
        await asyncio.sleep(0)
        assert expiry_owner.cancelled()

    @pytest.mark.asyncio
    async def test_terminate_untracked_returns_none(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        result = await lifecycle.terminate("ghost")
        assert result is None

    @pytest.mark.asyncio
    async def test_terminate_finalizes_authoritatively_already_gone_child(self):
        manager = _make_mock_manager()
        manager.terminate_child.return_value = False
        lifecycle = SpawnedAgentLifecycle(manager)
        await lifecycle.register(
            child_name="already-gone",
            child_did="did:child:already-gone",
            parent_did="did:parent",
            ttl_seconds=3600,
        )
        lifecycle._fire_hook = AsyncMock()

        result = await lifecycle.terminate("already-gone")

        assert result is not None
        assert result.status is SpawnStatus.TERMINATED
        assert result.finalized_from_absence is True
        assert not lifecycle.is_tracked("already-gone")
        lifecycle._fire_hook.assert_awaited_once()


class TestCascadingShutdown:
    """Parent termination terminates all children."""

    @pytest.mark.asyncio
    async def test_shutdown_terminates_all_children(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="child1",
            child_did="did:c1",
            parent_did="did:parent",
            ttl_seconds=3600,
        )
        await lifecycle.register(
            child_name="child2",
            child_did="did:c2",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        assert len(lifecycle.get_tracked_children()) == 2

        await lifecycle.shutdown()

        assert len(lifecycle.get_tracked_children()) == 0
        assert manager.terminate_child.await_count == 2

    @pytest.mark.asyncio
    async def test_shutdown_stores_results(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="child1",
            child_did="did:c1",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        await lifecycle.shutdown()

        result = lifecycle.get_result("child1")
        assert result is not None
        assert result.status == SpawnStatus.TERMINATED

    @pytest.mark.asyncio
    async def test_shutdown_refusal_keeps_tracking_ttl_and_no_terminal_result(self):
        from kestrel_sovereign.multi_agent.agent_manager import (
            ChildTerminationNotPerformedError,
        )

        manager = _make_mock_manager()
        manager.terminate_child.return_value = False
        manager.get_agent.return_value = object()
        manager.get_children.return_value = ["refused"]
        lifecycle = SpawnedAgentLifecycle(manager)
        await lifecycle.register(
            child_name="refused",
            child_did="did:refused",
            parent_did="did:parent",
            ttl_seconds=3600,
        )
        ttl_task = lifecycle._tracked["refused"].ttl_task
        lifecycle._fire_hook = AsyncMock()

        with pytest.raises(ChildTerminationNotPerformedError):
            await lifecycle.shutdown()

        assert lifecycle.is_tracked("refused")
        assert lifecycle.get_result("refused") is None
        assert lifecycle._tracked["refused"].ttl_task is ttl_task
        assert ttl_task is not None and not ttl_task.done()
        lifecycle._fire_hook.assert_not_awaited()
        ttl_task.cancel()
        await asyncio.gather(ttl_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_shutdown_continues_after_retained_offboarding(self):
        manager = _make_mock_manager()
        retained = RuntimeOffboardingRetainedError(
            agent_name="child1",
            agent_id="did:c1",
            runtime_path=Path("operator/runtime/child1"),
            cause=OSError("retained"),
        )
        manager.terminate_child.side_effect = [retained, True]
        lifecycle = SpawnedAgentLifecycle(manager)
        for child_name, child_did in (("child1", "did:c1"), ("child2", "did:c2")):
            await lifecycle.register(
                child_name=child_name,
                child_did=child_did,
                parent_did="did:parent",
                ttl_seconds=3600,
            )

        with pytest.raises(RuntimeOffboardingRetainedError):
            await lifecycle.shutdown()

        assert manager.terminate_child.await_count == 2
        assert lifecycle.get_tracked_children() == []


class TestHookEvents:
    """Hook events fire correctly for spawn and terminate."""

    @pytest.mark.asyncio
    async def test_spawn_fires_agent_spawn_hook(self):
        manager = _make_mock_manager()
        hooks = HooksManager()

        received_inputs = []

        class SpawnHook:
            def __init__(self):
                self.name = "test_spawn_hook"
                self.events = [HookEvent.AGENT_SPAWN]
                self.matcher = None
                self.priority = 100
                self.timeout = 5.0
                self.enabled = True
                self._compiled_matcher = None

            def matches(self, tool_name):
                return True

            async def execute(self, input: HookInput) -> HookOutput:
                received_inputs.append(input)
                return HookOutput.allow()

        hooks.register(SpawnHook())

        lifecycle = SpawnedAgentLifecycle(manager, hooks_manager=hooks)

        await lifecycle.register(
            child_name="hooked_child",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
            purpose="hook test",
        )

        assert len(received_inputs) == 1
        inp = received_inputs[0]
        assert inp.hook_event_name == "AgentSpawn"
        assert inp.parent_did == "did:parent"
        assert inp.child_did == "did:child"
        assert inp.child_name == "hooked_child"
        assert inp.spawn_purpose == "hook test"

        await lifecycle.shutdown()

    @pytest.mark.asyncio
    async def test_terminate_fires_agent_terminate_hook(self):
        manager = _make_mock_manager()
        hooks = HooksManager()

        received_inputs = []

        class TermHook:
            def __init__(self):
                self.name = "test_term_hook"
                self.events = [HookEvent.AGENT_TERMINATE]
                self.matcher = None
                self.priority = 100
                self.timeout = 5.0
                self.enabled = True
                self._compiled_matcher = None

            def matches(self, tool_name):
                return True

            async def execute(self, input: HookInput) -> HookOutput:
                received_inputs.append(input)
                return HookOutput.allow()

        hooks.register(TermHook())

        lifecycle = SpawnedAgentLifecycle(manager, hooks_manager=hooks)

        await lifecycle.register(
            child_name="hooked_child",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        await lifecycle.terminate("hooked_child", reason="done")

        assert len(received_inputs) == 1
        inp = received_inputs[0]
        assert inp.hook_event_name == "AgentTerminate"
        assert inp.child_name == "hooked_child"
        assert inp.termination_reason == "done"

    @pytest.mark.asyncio
    async def test_no_hooks_manager_no_error(self):
        """Lifecycle works fine without a HooksManager."""
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager, hooks_manager=None)

        await lifecycle.register(
            child_name="worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        # Should not raise
        await lifecycle.terminate("worker")

    @pytest.mark.asyncio
    async def test_ttl_fires_terminate_hook(self):
        """TTL expiry should also fire the AGENT_TERMINATE hook."""
        manager = _make_mock_manager()
        hooks = HooksManager()

        received = []

        class TermHook:
            def __init__(self):
                self.name = "ttl_term_hook"
                self.events = [HookEvent.AGENT_TERMINATE]
                self.matcher = None
                self.priority = 100
                self.timeout = 5.0
                self.enabled = True
                self._compiled_matcher = None

            def matches(self, tool_name):
                return True

            async def execute(self, input: HookInput) -> HookOutput:
                received.append(input)
                return HookOutput.allow()

        hooks.register(TermHook())

        lifecycle = SpawnedAgentLifecycle(manager, hooks_manager=hooks)

        await lifecycle.register(
            child_name="ttl_child",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=0.1,
        )

        await asyncio.sleep(0.3)

        assert len(received) == 1
        assert received[0].termination_reason == "TTL expired"


class TestHookEventEnum:
    """Verify the new HookEvent values exist."""

    def test_agent_spawn_event(self):
        assert HookEvent.AGENT_SPAWN.value == "AgentSpawn"

    def test_agent_terminate_event(self):
        assert HookEvent.AGENT_TERMINATE.value == "AgentTerminate"

    def test_hooks_manager_initializes_new_events(self):
        """HooksManager should have registries for the new events."""
        hooks = HooksManager()
        assert HookEvent.AGENT_SPAWN in hooks._hooks
        assert HookEvent.AGENT_TERMINATE in hooks._hooks
