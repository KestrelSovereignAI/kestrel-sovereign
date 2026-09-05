"""Unit tests for the in-process AgentManager."""

import asyncio
import json
import os
import sqlite3
import stat
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.testclient import TestClient

from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
from kestrel_sovereign.features import MandatoryFeatureReadinessError
from kestrel_sovereign.features.isolated_runtime import (
    IsolatedRuntimeNamespaceError,
    RuntimeNamespaceCleanupOutcome,
    derive_isolated_runtime_namespace,
    prepare_isolated_runtime_namespace,
    resolve_isolated_runtime_namespace,
)
from kestrel_sovereign.features.scheduler.runner import (
    AgentManagerHostedSchedulerExecutor,
    SchedulerExecution,
)
from kestrel_sovereign.identity.local_anchor import AgentDIDLookupMode
from kestrel_sovereign.identity.runtime_identity import IdentityReadinessError
from kestrel_sovereign.kestrel_agent import (
    KestrelAgent,
    arm_host_authority_deadline,
    await_lifecycle_task_completion,
)
from kestrel_sovereign.knowledge import InferenceError, InferenceProfile, OntologyRef
from kestrel_sovereign.multi_agent.agent_manager import (
    AgentOperationAdmission,
    AgentManager,
    ChildTerminationReconciliationError,
    PersistedSpawnParentUnavailableError,
    RUNTIME_OFFBOARD_TIMEOUT_S,
    RuntimeOffboardingAdmission,
    RuntimeOffboardingNotPerformedError,
    RuntimeOffboardingRetainedError,
    HostedIsolatedRuntimeLifecyclePolicy,
    _parse_runtime_offboard_timeout,
)
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.inception_service import generate_secp256k1_keypair
from kestrel_sovereign.spawn.authority_registry import SpawnAuthorityRegistry
from kestrel_sovereign.spawn.mandate import (
    PersistedSpawnMandateExpiredError,
    SpawnMandate,
    remaining_spawn_ttl_seconds,
    sign_mandate,
    verify_mandate,
)
from kestrel_sovereign.spawn.mandate_reload import read_spawn_mandate
from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle
from kestrel_sovereign.signals import OrderedLockManager
from tests.utils.aiosqlite_workers import aiosqlite_worker


def _make_mock_agent(agent_id: str = "did:pkh:eip155:1:0xABC"):
    """Create a mock KestrelAgent with the minimum interface."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.initialize = AsyncMock()
    agent.shutdown = AsyncMock()
    agent.get_agent_card = AsyncMock()
    return agent


async def _persist_and_publish_spawn_test_child(
    manager: AgentManager,
    name: str,
    child,
    spawn_kwargs: dict,
) -> None:
    """Model create -> load's signed-receipt-before-routing contract."""

    if vars(child).get("_raw_storage") is None:
        child._raw_storage = SimpleNamespace(
            graph=SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
        )
    admission = manager._agent_operations[manager._canonical_agent_name(name)]
    assert admission.before_publish is not None
    if admission.spawn_candidate_config is None:
        admission.spawn_candidate_config = LocalAgentConfig(
            data_dir=Path("agent_data") / name,
            port=8802,
        )
    if admission.spawn_authority_pending_id is None:
        pending = manager._spawn_authority_registry.reserve_pending(
            child_name=name,
            parent_did=spawn_kwargs["parent_did"],
            mandate=spawn_kwargs["mandate"],
            config=admission.spawn_candidate_config,
        )
        admission.spawn_authority_pending_id = pending.reservation_id
    assert manager.get_agent(name) is None
    await admission.before_publish(child)
    manager._agents[name] = child
    manager._agent_names[child.agent_id] = name


async def _load_spawn_after_mocked_inception(
    manager: AgentManager,
    name: str,
    kwargs: dict,
    *,
    config: LocalAgentConfig | None = None,
):
    """Model create_agent's pre-inception denial before exercising real load."""

    config = config or LocalAgentConfig(data_dir="unused", port=8801)
    admission = manager._agent_operations[name.casefold()]
    admission.spawn_candidate_config = config
    pending = manager._spawn_authority_registry.reserve_pending(
        child_name=name,
        parent_did=kwargs["parent_did"],
        mandate=kwargs["mandate"],
        config=config,
    )
    admission.spawn_authority_pending_id = pending.reservation_id
    return await manager.load_agent(name, config)


def _signed_restored_mandate(
    parent_did: str,
    child_did: str,
    **kwargs,
) -> tuple[MagicMock, SpawnMandate]:
    private_key, _ = generate_secp256k1_keypair()
    parent = _make_mock_agent(parent_did)
    parent._private_key = private_key
    parent.identity = None
    parent._persisted_spawn_mandate = None
    mandate = SpawnMandate(parent_did=parent_did, child_did=child_did, **kwargs)
    return parent, sign_mandate(mandate, private_key)


@pytest.mark.asyncio
async def test_signed_receipt_round_trip_preserves_integer_budget_signature():
    private_key, public_key = generate_secp256k1_keypair()
    mandate = sign_mandate(
        SpawnMandate(
            parent_did="did:parent-int-budget",
            child_did="did:child-int-budget",
            budget_allocation=1,
        ),
        private_key,
    )
    edge = SimpleNamespace(
        label="spawned_by",
        source_id=mandate.child_did,
        target_id=mandate.parent_did,
        properties=mandate.to_edge_properties(),
    )
    storage = SimpleNamespace(get_edges_from=AsyncMock(return_value=[edge]))

    restored = await read_spawn_mandate(storage, mandate.child_did)

    assert restored is not None
    assert type(restored.budget_allocation) is int
    assert verify_mandate(restored, public_key)


@pytest.mark.asyncio
async def test_spawn_refuses_mandate_for_a_different_parent(tmp_path):
    manager = AgentManager(base_data_dir=tmp_path)
    parent = _make_mock_agent("did:actual-parent")

    with pytest.raises(ValueError, match="parent DID"):
        await manager.spawn_agent(
            "Child",
            parent,
            SpawnMandate(parent_did="did:other-parent"),
        )

    assert manager._agent_operations == {}


@pytest.mark.asyncio
async def test_spawn_refuses_prebound_child_identity(tmp_path):
    manager = AgentManager(base_data_dir=tmp_path)
    parent = _make_mock_agent("did:actual-parent")

    with pytest.raises(ValueError, match="child DID must be unset"):
        await manager.spawn_agent(
            "Child",
            parent,
            SpawnMandate(
                parent_did=parent.agent_id,
                child_did="did:preselected-child",
            ),
        )

    assert manager._agent_operations == {}


@pytest.mark.asyncio
async def test_spawn_snapshots_mutable_mandate_before_admission_await(tmp_path):
    """Caller mutation cannot change authority after proposal validation."""

    manager = AgentManager(base_data_dir=tmp_path)
    parent = _make_mock_agent("did:test:snapshot-parent")
    parent.features = {"AllowedFeature": SimpleNamespace()}
    original = SpawnMandate(
        parent_did=parent.agent_id,
        features_allowed=["AllowedFeature"],
        additional_constraints={
            "restricted_tool_args": {"fetch": {"host": ["example.test"]}}
        },
        purpose="validated proposal",
    )
    admitted = asyncio.Event()
    release_admission = asyncio.Event()
    real_admit = manager._admit_agent_operation

    async def blocked_admit(name, *, kind):
        result = await real_admit(name, kind=kind)
        admitted.set()
        await release_admission.wait()
        return result

    child = _make_mock_agent("did:test:snapshot-child")
    manager._admit_agent_operation = blocked_admit
    manager._run_admitted_spawn = AsyncMock(return_value=child)
    spawn = asyncio.create_task(
        manager.spawn_agent("SnapshotChild", parent, original)
    )
    await asyncio.wait_for(admitted.wait(), timeout=1.0)

    original.parent_did = "did:test:mutated-parent"
    original.features_allowed.append("EscalatedFeature")
    original.additional_constraints["restricted_tool_args"]["fetch"][
        "host"
    ].append("mutated.test")
    original.purpose = "mutated after validation"
    release_admission.set()

    assert await spawn is child
    captured = manager._run_admitted_spawn.await_args.args[2]
    assert captured is not original
    assert captured.parent_did == parent.agent_id
    assert captured.features_allowed == ["AllowedFeature"]
    assert captured.additional_constraints == {
        "restricted_tool_args": {"fetch": {"host": ["example.test"]}}
    }
    assert captured.purpose == "validated proposal"
    await manager._release_agent_operations(
        [
            manager._agent_operations[
                manager._canonical_agent_name("SnapshotChild")
            ]
        ]
    )


@pytest.mark.asyncio
async def test_load_awaits_spawn_receipt_before_routing_publication(tmp_path):
    manager = AgentManager(base_data_dir=tmp_path)
    child = _make_mock_agent("did:test:prepublication-child")
    manager._initialize_agent = AsyncMock(return_value=child)
    manager._on_agent_registered = AsyncMock()
    admission, owns = await manager._admit_agent_operation(
        "PrepublicationChild",
        kind="spawn",
    )
    assert owns
    observed: list[object] = []

    async def persist_before_publish(candidate):
        observed.append(manager.get_agent("PrepublicationChild"))
        observed.append(candidate)

    async def run_ready(candidate):
        observed.append(
            ("ready", candidate, manager.get_agent("PrepublicationChild"))
        )

    admission.before_publish = persist_before_publish
    manager._run_hosted_agent_ready_hooks = AsyncMock(side_effect=run_ready)
    try:
        loaded = await manager.load_agent(
            "PrepublicationChild",
            LocalAgentConfig(data_dir="unused", port=8801),
        )
    finally:
        await manager._release_agent_operation(admission)

    assert loaded is child
    assert observed == [None, child, ("ready", child, child)]
    assert manager.get_agent("PrepublicationChild") is child


@pytest.mark.asyncio
async def test_spawn_receipt_wait_cannot_outlive_one_admission_deadline(tmp_path):
    """A prepublication writer cannot keep a signed child alive past expiry."""

    parent, mandate = _signed_restored_mandate(
        "did:test:deadline-parent",
        "did:test:deadline-child",
        ttl_seconds=1,
        created_at=(
            datetime.now(timezone.utc) - timedelta(seconds=0.8)
        ).isoformat(),
    )
    child = _make_mock_agent(mandate.child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("DeadlineParent", parent)
    manager._initialize_agent = AsyncMock(return_value=child)
    manager._run_hosted_agent_ready_hooks = AsyncMock()
    admission, owns = await manager._admit_agent_operation(
        "DeadlineChild",
        kind="spawn",
    )
    assert owns
    never_persisted = asyncio.Event()

    async def blocked_receipt(_candidate):
        await never_persisted.wait()

    admission.before_publish = blocked_receipt
    try:
        with pytest.raises(RuntimeError, match="prepublication authority"):
            await asyncio.wait_for(
                manager.load_agent(
                    "DeadlineChild",
                    LocalAgentConfig(data_dir="unused", port=8801),
                ),
                timeout=1.0,
            )
    finally:
        never_persisted.set()
        await manager._release_agent_operation(admission)

    assert manager.get_agent("DeadlineChild") is None
    manager._run_hosted_agent_ready_hooks.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_acquire_admission_expiry_releases_a2a_writer(tmp_path):
    manager = AgentManager(base_data_dir=tmp_path)

    async def expire_after_acquire(operation, **_kwargs):
        await operation()
        raise RuntimeError("expired after acquiring publication writer")

    manager._await_before_spawn_admission_deadline = expire_after_acquire

    with pytest.raises(RuntimeError, match="expired after acquiring"):
        async with manager._a2a_writer_before_spawn_admission_deadline(
            deadline=1.0,
            phase="test publication",
        ):
            pytest.fail("expired writer must not enter its protected body")

    await asyncio.wait_for(manager._a2a_lifecycle_lock.acquire(), timeout=0.1)
    manager._a2a_lifecycle_lock.release()

@pytest.mark.asyncio
async def test_load_validates_restored_authority_before_agent_ready(tmp_path):
    """A rejected cold child must never cross the wake-capable ready boundary."""

    parent_did = "did:pkh:eip155:1:0xReadyParent"
    child_did = "did:pkh:eip155:1:0xReadyChild"
    parent, mandate = _signed_restored_mandate(parent_did, child_did)
    mandate.purpose = "tampered after signing"
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("ReadyParent", parent)
    ready_events: list[str] = []
    active_initialization_events: list[str] = []

    class HostedChild:
        def __init__(self, *, did, **_kwargs):
            self.agent_id = did
            self.did = did
            self.identity = None
            self._persisted_spawn_mandate = mandate

        async def initialize(self):
            await self._host_authority_preflight(self)
            active_initialization_events.append("features initialized")
            if not vars(self).get("_host_ready_hooks_deferred", False):
                ready_events.append("ready")

        async def run_agent_ready_hooks(self):
            ready_events.append("ready")

        async def shutdown(self):
            return None

    config = LocalAgentConfig(data_dir=Path("ready-child"), port=8801)
    with (
        patch.object(LocalAgentConfig, "validate_runtime", return_value=[]),
        patch(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            new=AsyncMock(return_value=child_did),
        ),
        patch(
            "kestrel_sovereign.multi_agent.agent_manager.KestrelAgent",
            HostedChild,
        ),
        pytest.raises(RuntimeError, match="signature is invalid"),
    ):
        await manager.load_agent("ReadyChild", config)

    assert ready_events == []
    assert active_initialization_events == []
    assert manager.get_agent("ReadyChild") is None


@pytest.mark.asyncio
async def test_load_projects_restored_authority_before_ready_and_rolls_it_back(
    tmp_path,
):
    """Wake-capable hooks run only inside a reserved authority boundary."""

    parent_did = "did:pkh:eip155:1:0xPreparedParent"
    child_did = "did:pkh:eip155:1:0xPreparedChild"
    parent, mandate = _signed_restored_mandate(parent_did, child_did)
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("PreparedParent", parent)
    manager._initialize_agent = AsyncMock(return_value=child)
    observed_authority = []

    async def fail_after_observing_authority(_agent):
        observed_authority.append(manager.get_mandate("PreparedChild"))
        observed_authority.append(_agent._agent_manager)
        raise RuntimeError("ready hook failed after authority observation")

    manager._run_hosted_agent_ready_hooks = AsyncMock(
        side_effect=fail_after_observing_authority
    )

    with pytest.raises(RuntimeError, match="ready hook failed"):
        await manager.load_agent(
            "PreparedChild",
            LocalAgentConfig(data_dir="unused", port=8801),
        )

    assert observed_authority == [mandate, manager]
    assert manager.get_agent("PreparedChild") is None
    assert manager.get_mandate("PreparedChild") is None
    assert manager.get_children(parent_did) == []
    assert child._agent_manager is None
    assert manager._prepared_agent_names == {}


@pytest.mark.asyncio
async def test_spawned_by_readiness_rollback_offboards_rejected_runtime_credentials(
    tmp_path,
):
    """A published-but-rejected spawn must not retain its private runtime."""

    manager = AgentManager(base_data_dir=tmp_path)
    did = "did:pkh:spawn-readiness-rollback"
    scope = resolve_isolated_runtime_namespace(
        manager._isolated_runtime_root,
        derive_isolated_runtime_namespace(did),
    )
    prepare_isolated_runtime_namespace(scope, did)
    credential = scope.path / "credential"
    credential.write_text("must-not-survive-rejected-readiness")
    child = _make_mock_agent(did)
    child.did = did
    child.isolated_runtime_scope = scope
    admission, owns = await manager._admit_agent_operation(
        "RejectedReadyChild",
        kind="spawn",
    )
    assert owns
    admission.child = child
    admission.published = True
    manager._agents["RejectedReadyChild"] = child
    manager._agent_names[did] = "RejectedReadyChild"
    manager._downgrade_uncommitted_spawn_receipt = AsyncMock()
    try:
        await manager._rollback_published_agent_after_readiness_failure(
            "RejectedReadyChild",
            child,
            admission,
        )

        assert manager.get_agent("RejectedReadyChild") is None
        assert not scope.path.exists()
    finally:
        await manager._release_agent_operation(admission)


@pytest.mark.asyncio
async def test_ready_failure_cascades_published_descendant_before_parent_rollback(
    tmp_path,
):
    """Post-publication readiness rollback cannot strand a routed orphan."""

    parent = _make_mock_agent("did:test:published-ready-parent")
    descendant = _make_mock_agent("did:test:published-ready-descendant")
    manager = AgentManager(base_data_dir=tmp_path)
    manager._initialize_agent = AsyncMock(return_value=parent)
    manager._on_agent_registered = AsyncMock()
    ready_observations: list[object] = []

    async def publish_descendant_then_fail(candidate):
        ready_observations.append(manager.get_agent("PublishedReadyParent"))
        manager._register_agent("ReadyDescendant", descendant)
        manager._parent_children[candidate.agent_id] = ["ReadyDescendant"]
        manager._child_mandates["ReadyDescendant"] = SpawnMandate(
            parent_did=candidate.agent_id,
            child_did=descendant.agent_id,
        )
        raise RuntimeError("ready hook failed after descendant publication")

    manager._run_hosted_agent_ready_hooks = AsyncMock(
        side_effect=publish_descendant_then_fail
    )

    with pytest.raises(RuntimeError, match="ready hook failed"):
        await manager.load_agent(
            "PublishedReadyParent",
            LocalAgentConfig(data_dir="unused", port=8801),
        )

    assert ready_observations == [parent]
    manager._on_agent_registered.assert_awaited_once_with(
        "PublishedReadyParent", parent
    )
    assert manager.get_agent("PublishedReadyParent") is None
    assert manager.get_agent("ReadyDescendant") is None
    assert manager.get_children(parent.agent_id) == []
    descendant.shutdown.assert_awaited_once()
    parent.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_spawn_installs_budget_custody_before_ready_hook(tmp_path):
    """A wake-capable child never observes its unrestricted boot wallet."""

    parent = _make_mock_agent("did:test:budget-ready-parent")
    parent._private_key, _ = generate_secp256k1_keypair()
    parent.identity = None
    parent.features = {}

    class ParentWallet:
        _balances = {}

        @staticmethod
        def can_afford(_amount, _currency):
            return True

    parent.wallet = ParentWallet()
    child = _make_mock_agent("did:test:budget-ready-child")
    child._raw_storage = SimpleNamespace(
        graph=SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
    )
    delegated = SimpleNamespace(
        allocation=SimpleNamespace(child_did=child.agent_id),
        refund_to_parent=AsyncMock(return_value=Decimal("5")),
        spent=Decimal("0"),
    )
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("BudgetReadyParent", parent)
    manager._initialize_agent = AsyncMock(return_value=child)
    manager._on_agent_registered = AsyncMock()
    events: list[str] = []

    async def create_through_real_load(name, **kwargs):
        return await _load_spawn_after_mocked_inception(manager, name, kwargs)

    async def observe_custody(candidate):
        events.append("ready")
        assert candidate.wallet is delegated
        assert candidate.wallet_agent is delegated
        assert candidate._delegated_wallet is delegated

    manager.create_agent = AsyncMock(side_effect=create_through_real_load)
    manager._run_hosted_agent_ready_hooks = AsyncMock(
        side_effect=observe_custody
    )

    with patch(
        "kestrel_sovereign.multi_agent.agent_manager.create_delegated_wallet",
        new=AsyncMock(return_value=delegated),
    ) as provision:
        spawned = await manager.spawn_agent(
            "BudgetReadyChild",
            parent,
            SpawnMandate(
                parent_did=parent.agent_id,
                budget_allocation=5,
                ttl_seconds=60,
            ),
        )

    assert spawned is child
    assert events == ["ready"]
    provision.assert_awaited_once()
    assert await manager.terminate_child(
        parent.agent_id,
        "BudgetReadyChild",
    ) is True


@pytest.mark.asyncio
async def test_cold_ready_failure_rolls_back_under_owned_scheduler_lock(tmp_path):
    """Cold-wake rollback reuses, rather than reacquires, its DID writer."""

    agent_id = "did:test:cold-ready-failure"
    config = LocalAgentConfig(data_dir="unused", port=8801, autostart=False)
    candidate = _make_mock_agent(agent_id)
    manager = AgentManager(base_data_dir=tmp_path)
    manager._seed_scheduler_authority({agent_id: ("ColdReady", config)})
    manager._initialize_agent = AsyncMock(return_value=candidate)
    manager._on_agent_registered = AsyncMock()
    manager._run_hosted_agent_ready_hooks = AsyncMock(
        side_effect=RuntimeError("cold ready failed")
    )

    lifecycle_lock = manager.scheduler_lifecycle_lock(agent_id)
    async with lifecycle_lock:
        with pytest.raises(RuntimeError, match="cold ready failed"):
            await asyncio.wait_for(
                manager.load_agent(
                    "ColdReady",
                    config,
                    expected_agent_id=agent_id,
                    scheduler_lifecycle_lock_held=True,
                ),
                timeout=0.5,
            )
        assert lifecycle_lock.locked()

    assert manager.get_agent("ColdReady") is None
    candidate.shutdown.assert_awaited_once()
    assert manager.scheduler_authority_for(agent_id) == ("ColdReady", config)
    assert manager.is_scheduler_agent_authorized(agent_id)

    retry = _make_mock_agent(agent_id)
    manager._initialize_agent = AsyncMock(return_value=retry)
    manager._run_hosted_agent_ready_hooks = AsyncMock()
    async with lifecycle_lock:
        loaded = await manager.load_agent(
            "ColdReady",
            config,
            expected_agent_id=agent_id,
            scheduler_lifecycle_lock_held=True,
        )
    assert loaded is retry


@pytest.mark.asyncio
async def test_ready_failure_runs_host_onboarding_rollback(tmp_path):
    """Host mutations remain provisional until readiness has committed."""

    candidate = _make_mock_agent("did:test:host-ready-rollback")
    manager = AgentManager(base_data_dir=tmp_path)
    manager._initialize_agent = AsyncMock(return_value=candidate)
    host_state: list[str] = []
    rollback = AsyncMock(side_effect=lambda: host_state.remove("HostReady"))

    async def onboard(name, _agent):
        host_state.append(name)
        return rollback

    manager.set_agent_registration_hook(onboard)
    manager._run_hosted_agent_ready_hooks = AsyncMock(
        side_effect=RuntimeError("ready failed")
    )

    with pytest.raises(RuntimeError, match="ready failed"):
        await manager.load_agent(
            "HostReady",
            LocalAgentConfig(data_dir="unused", port=8801),
        )

    assert host_state == []
    rollback.assert_awaited_once()
    assert manager.get_agent("HostReady") is None


@pytest.mark.asyncio
async def test_spawn_ready_failure_revokes_receipt_before_child_shutdown(tmp_path):
    """A rejected published child keeps its graph open through revocation."""

    parent = _make_mock_agent("did:test:ready-receipt-parent")
    parent._private_key, _ = generate_secp256k1_keypair()
    parent.identity = None
    parent.features = {}

    class ReceiptGraph:
        def __init__(self):
            self.closed = False
            self.writes: list[dict] = []

        async def add_trusted_cross_agent_edge(
            self, _source, _target, _relationship, *, properties
        ):
            if self.closed:
                raise RuntimeError("receipt graph was closed before revocation")
            self.writes.append(dict(properties))

    graph = ReceiptGraph()
    child = _make_mock_agent("did:test:ready-receipt-child")
    child._raw_storage = SimpleNamespace(graph=graph)

    async def close_receipt_graph():
        graph.closed = True

    child.shutdown = AsyncMock(side_effect=close_receipt_graph)
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("ReceiptReadyParent", parent)
    manager._initialize_agent = AsyncMock(return_value=child)

    async def create_through_real_load(name, **kwargs):
        return await _load_spawn_after_mocked_inception(manager, name, kwargs)

    manager.create_agent = AsyncMock(side_effect=create_through_real_load)
    manager._run_hosted_agent_ready_hooks = AsyncMock(
        side_effect=RuntimeError("ready receipt failed")
    )

    with pytest.raises(RuntimeError, match="ready receipt failed"):
        await manager.spawn_agent(
            "ReceiptReadyChild",
            parent,
            SpawnMandate(parent_did=parent.agent_id, ttl_seconds=60),
        )

    assert len(graph.writes) == 2
    assert graph.writes[0]["parent_signature"]
    assert graph.writes[1]["parent_signature"] is None
    assert child.shutdown.await_count == 1
    assert manager.get_agent("ReceiptReadyChild") is None


@pytest.mark.asyncio
async def test_post_ready_spawn_failure_destructively_cascades_descendant_before_rollback(
    tmp_path,
):
    """Later governance failure cannot retain work created by ready hooks."""

    parent = _make_mock_agent("did:test:post-ready-parent")
    parent._private_key, _ = generate_secp256k1_keypair()
    parent.identity = None
    parent.features = {}
    child = _make_mock_agent("did:test:post-ready-child")
    child._raw_storage = SimpleNamespace(
        graph=SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
    )
    descendant = _make_mock_agent("did:test:post-ready-descendant")
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("PostReadyParent", parent)
    manager._initialize_agent = AsyncMock(return_value=child)
    manager._on_agent_registered = AsyncMock()

    async def create_through_real_load(name, **kwargs):
        return await _load_spawn_after_mocked_inception(manager, name, kwargs)

    async def publish_descendant(candidate):
        manager._register_agent("PostReadyDescendant", descendant)
        manager._parent_children[candidate.agent_id] = [
            "PostReadyDescendant"
        ]
        manager._child_mandates["PostReadyDescendant"] = SpawnMandate(
            parent_did=candidate.agent_id,
            child_did=descendant.agent_id,
        )

    manager.create_agent = AsyncMock(side_effect=create_through_real_load)
    manager._run_hosted_agent_ready_hooks = AsyncMock(
        side_effect=publish_descendant
    )
    manager._ensure_spawn_operation_admitted = AsyncMock(
        side_effect=RuntimeError("post-ready governance failed")
    )
    real_terminate_children = manager.terminate_children
    manager.terminate_children = AsyncMock(wraps=real_terminate_children)

    with pytest.raises(RuntimeError, match="post-ready governance failed"):
        await manager.spawn_agent(
            "PostReadyChild",
            parent,
            SpawnMandate(parent_did=parent.agent_id),
        )

    assert manager.get_agent("PostReadyChild") is None
    assert manager.get_agent("PostReadyDescendant") is None
    assert manager.get_children(child.agent_id) == []
    assert any(
        args == (child.agent_id,) and kwargs.get("offboard_runtime") is True
        for args, kwargs in manager.terminate_children.await_args_list
    )
    descendant.shutdown.assert_awaited_once()
    child.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_destructive_child_termination_removes_spawn_startup_row(tmp_path):
    """A destructively terminated child is absent from the next host roster."""

    child_name = "RosterChild"
    parent_did = "did:test:roster-parent"
    child_did = "did:test:roster-child"
    child_config = LocalAgentConfig(
        data_dir=Path("agent_data") / child_name,
        port=8802,
    )
    config_path = tmp_path / "multi_agent.toml"
    MultiAgentConfig(agents={child_name: child_config}).save(config_path)
    child_dir = tmp_path / child_config.data_dir
    child_dir.mkdir(parents=True)
    (child_dir / "kestrel_prime.db").touch()

    manager = AgentManager(
        base_data_dir=tmp_path,
        startup_config_path=config_path,
    )
    child = _make_mock_agent(child_did)
    child.did = child_did
    scope = resolve_isolated_runtime_namespace(
        manager._isolated_runtime_root,
        derive_isolated_runtime_namespace(child_did),
    )
    prepare_isolated_runtime_namespace(scope, child_did)
    child.isolated_runtime_scope = scope
    manager._agents[child_name] = child
    manager._agent_names[child_did] = child_name
    manager._created_configs[child_name] = child_config
    manager._parent_children[parent_did] = [child_name]
    manager._child_mandates[child_name] = SpawnMandate(
        parent_did=parent_did,
        child_did=child_did,
    )

    assert await manager.terminate_child(
        parent_did,
        child_name,
        offboard_runtime=True,
    )

    persisted = MultiAgentConfig.from_file(config_path)
    assert child_name not in persisted.agents
    assert not scope.path.exists()


@pytest.mark.asyncio
async def test_destructive_storage_child_termination_retires_auto_discovery(
    tmp_path,
):
    """A terminal NOT_HOSTED outcome cannot resurrect a removed child."""

    child_name = "StorageRosterChild"
    parent_did = "did:test:storage-roster-parent"
    child_did = "did:test:storage-roster-child"
    agent_data = tmp_path / "agent_data"
    child_dir = agent_data / child_name
    child_dir.mkdir(parents=True)
    (child_dir / "kestrel_prime.db").touch()

    manager = AgentManager(base_data_dir=tmp_path)
    child = _make_mock_agent(child_did)
    child.did = child_did
    manager._agents[child_name] = child
    manager._agent_names[child_did] = child_name
    manager._created_configs[child_name] = LocalAgentConfig(
        data_dir=Path("agent_data") / child_name,
        port=8802,
    )
    manager._parent_children[parent_did] = [child_name]
    manager._child_mandates[child_name] = SpawnMandate(
        parent_did=parent_did,
        child_did=child_did,
    )

    with pytest.raises(RuntimeOffboardingNotPerformedError) as raised:
        await manager.terminate_child(
            parent_did,
            child_name,
            offboard_runtime=True,
        )

    assert raised.value.metadata["runtime_cleanup_state"] == "not_hosted"
    assert manager.get_agent(child_name) is None
    assert manager.get_children(parent_did) == []
    assert (child_dir / ".kestrel-spawn-retired").read_text().strip() == child_did
    assert child_name not in MultiAgentConfig.auto_discover(agent_data).agents


@pytest.mark.asyncio
async def test_refused_child_offboarding_restores_spawn_startup_row(tmp_path):
    """Desired state is compensated when destructive cleanup never starts."""

    child_name = "RetainedRosterChild"
    parent_did = "did:test:retained-roster-parent"
    child_did = "did:test:retained-roster-child"
    child_config = LocalAgentConfig(
        data_dir=Path("agent_data") / child_name,
        port=8802,
    )
    config_path = tmp_path / "multi_agent.toml"
    MultiAgentConfig(agents={child_name: child_config}).save(config_path)
    manager = AgentManager(
        base_data_dir=tmp_path,
        startup_config_path=config_path,
    )
    child = _make_mock_agent(child_did)
    child.did = child_did
    child.shutdown.side_effect = RuntimeError("shutdown refused")
    manager._agents[child_name] = child
    manager._agent_names[child_did] = child_name
    manager._created_configs[child_name] = child_config
    manager._parent_children[parent_did] = [child_name]
    manager._child_mandates[child_name] = SpawnMandate(
        parent_did=parent_did,
        child_did=child_did,
    )

    assert not await manager.terminate_child(
        parent_did,
        child_name,
        offboard_runtime=True,
    )

    persisted = MultiAgentConfig.from_file(config_path)
    assert persisted.agents[child_name] == child_config
    assert manager.get_agent(child_name) is child


@pytest.mark.asyncio
async def test_ready_hook_removal_cannot_return_withdrawn_agent_as_loaded(tmp_path):
    candidate = _make_mock_agent("did:test:ready-self-removed")
    manager = AgentManager(base_data_dir=tmp_path)
    manager._initialize_agent = AsyncMock(return_value=candidate)
    manager._on_agent_registered = AsyncMock()

    async def remove_during_ready(_candidate):
        assert await manager.remove_agent("ReadySelfRemoved") is True

    manager._run_hosted_agent_ready_hooks = AsyncMock(
        side_effect=remove_during_ready
    )

    with pytest.raises(RuntimeError, match="withdrawn during readiness"):
        await manager.load_agent(
            "ReadySelfRemoved",
            LocalAgentConfig(data_dir="unused", port=8801),
        )

    assert manager.get_agent("ReadySelfRemoved") is None
    candidate.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_spawn_ready_hook_self_removal_revokes_receipt_before_shutdown(tmp_path):
    """The spawn owner keeps receipt custody through the governance commit."""

    parent = _make_mock_agent("did:test:ready-self-remove-parent")
    parent._private_key, _ = generate_secp256k1_keypair()
    parent.identity = None
    parent.features = {}

    class ReceiptGraph:
        def __init__(self):
            self.closed = False
            self.writes: list[dict] = []

        async def add_trusted_cross_agent_edge(
            self, _source, _target, _relationship, *, properties
        ):
            if self.closed:
                raise RuntimeError("receipt graph was closed before revocation")
            self.writes.append(dict(properties))

    graph = ReceiptGraph()
    child = _make_mock_agent("did:test:ready-self-remove-child")
    child.features = {}
    child._raw_storage = SimpleNamespace(graph=graph)

    async def close_receipt_graph():
        graph.closed = True

    child.shutdown = AsyncMock(side_effect=close_receipt_graph)
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("ReadySelfRemoveParent", parent)
    manager._initialize_agent = AsyncMock(return_value=child)

    async def create_through_real_load(name, **kwargs):
        return await _load_spawn_after_mocked_inception(manager, name, kwargs)

    async def remove_during_ready(_candidate):
        await manager.remove_agent("ReadySelfRemoveChild")

    manager.create_agent = AsyncMock(side_effect=create_through_real_load)
    manager._run_hosted_agent_ready_hooks = AsyncMock(
        side_effect=remove_during_ready
    )

    with pytest.raises(RuntimeError, match="governance commit"):
        await manager.spawn_agent(
            "ReadySelfRemoveChild",
            parent,
            SpawnMandate(parent_did=parent.agent_id),
        )

    assert len(graph.writes) == 2
    assert graph.writes[0]["parent_signature"]
    assert graph.writes[1]["parent_signature"] is None
    assert child.shutdown.await_count == 1
    assert manager.get_agent("ReadySelfRemoveChild") is None


@pytest.mark.asyncio
async def test_dynamic_child_waits_for_initializing_parent_publication(tmp_path):
    """A verified parent may finish its own publication after its child boots."""

    parent_did = "did:pkh:eip155:1:0xConcurrentRestoreParent"
    child_did = "did:pkh:eip155:1:0xConcurrentRestoreChild"
    parent, child_mandate = _signed_restored_mandate(parent_did, child_did)
    parent.features = {}
    child = _make_mock_agent(child_did)
    child.features = {}
    child._persisted_spawn_mandate = child_mandate
    candidates = {"ConcurrentParent": parent, "ConcurrentChild": child}

    manager = AgentManager(base_data_dir=tmp_path)
    parent_initializing = asyncio.Event()
    child_initialized = asyncio.Event()
    release_parent_initialization = asyncio.Event()

    async def initialize(name, _config, **_kwargs):
        candidate = candidates[name]
        manager._initializing_agents[name] = candidate
        candidate._agent_manager_authority_evidence_loaded = True
        candidate._agent_manager_authority_evidence_event = asyncio.Event()
        candidate._agent_manager_authority_evidence_event.set()
        candidate._agent_manager_published = False
        candidate._agent_manager_publication_event = asyncio.Event()
        if name == "ConcurrentParent":
            parent_initializing.set()
            await release_parent_initialization.wait()
        else:
            child_initialized.set()
        return candidate

    manager._initialize_agent = AsyncMock(side_effect=initialize)
    manager._run_host_onboarding_before_mandate_expiry = AsyncMock()
    manager._finish_published_agent_readiness = AsyncMock()

    parent_load = asyncio.create_task(
        manager.load_agent(
            "ConcurrentParent",
            LocalAgentConfig(data_dir="parent", port=8801),
        )
    )
    await parent_initializing.wait()
    child_load = asyncio.create_task(
        manager.load_agent(
            "ConcurrentChild",
            LocalAgentConfig(data_dir="child", port=8802),
        )
    )
    await child_initialized.wait()
    await asyncio.sleep(0)
    assert not child_load.done()

    release_parent_initialization.set()
    loaded_parent, loaded_child = await asyncio.gather(parent_load, child_load)

    assert loaded_parent is parent
    assert loaded_child is child
    assert manager.get_agent("ConcurrentParent") is parent
    assert manager.get_agent("ConcurrentChild") is child


@pytest.mark.asyncio
async def test_descendant_cleanup_error_still_withdraws_rejected_ready_parent(
    tmp_path,
):
    candidate = _make_mock_agent("did:test:ready-cleanup-parent")
    manager = AgentManager(base_data_dir=tmp_path)
    manager._initialize_agent = AsyncMock(return_value=candidate)
    manager._on_agent_registered = AsyncMock()
    manager._run_hosted_agent_ready_hooks = AsyncMock(
        side_effect=RuntimeError("ready failed")
    )
    manager.terminate_children = AsyncMock(
        side_effect=ChildTerminationReconciliationError(
            child_name="removed-descendant",
            cause=RuntimeError("refund failed"),
        )
    )

    with pytest.raises(ExceptionGroup, match="published rollback failed"):
        await manager.load_agent(
            "ReadyCleanupParent",
            LocalAgentConfig(data_dir="unused", port=8801),
        )

    manager.terminate_children.assert_awaited_once_with(candidate.agent_id)
    assert manager.get_agent("ReadyCleanupParent") is None
    candidate.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_prepared_leaf_cannot_spawn_from_ready_hook_before_publication(
    tmp_path,
):
    """A private verified parent keeps its depth ceiling during readiness."""

    root_did = "did:pkh:eip155:1:0xPreparedDepthRoot"
    leaf_did = "did:pkh:eip155:1:0xPreparedDepthLeaf"
    root, leaf_mandate = _signed_restored_mandate(
        root_did,
        leaf_did,
        max_child_depth=0,
    )
    leaf = _make_mock_agent(leaf_did)
    leaf.features = {}
    leaf._persisted_spawn_mandate = leaf_mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("PreparedDepthRoot", root)
    manager._initialize_agent = AsyncMock(return_value=leaf)
    manager._do_spawn = AsyncMock(
        side_effect=AssertionError("prepared leaf bypassed its depth ceiling")
    )
    observed: list[BaseException] = []

    async def attempt_grandchild(candidate):
        try:
            await manager.spawn_agent(
                "ForbiddenGrandchild",
                candidate,
                SpawnMandate(parent_did=candidate.agent_id),
            )
        except BaseException as exc:
            observed.append(exc)

    manager._run_hosted_agent_ready_hooks = AsyncMock(
        side_effect=attempt_grandchild
    )

    loaded = await manager.load_agent(
        "PreparedDepthLeaf",
        LocalAgentConfig(data_dir="unused", port=8801),
    )

    assert loaded is leaf
    assert len(observed) == 1
    assert isinstance(observed[0], ValueError)
    assert "max child depth" in str(observed[0])
    manager._do_spawn.assert_not_awaited()
    assert manager._prepared_agent_names == {}


@pytest.mark.asyncio
async def test_live_spawn_projects_signed_leaf_authority_before_publication(
    tmp_path,
):
    """A fresh signed child cannot use its provisional window to spawn."""

    root_did = "did:pkh:eip155:1:0xLiveDepthRoot"
    leaf_did = "did:pkh:eip155:1:0xLiveDepthLeaf"
    root, leaf_mandate = _signed_restored_mandate(
        root_did,
        leaf_did,
        max_child_depth=0,
    )
    leaf = _make_mock_agent(leaf_did)
    leaf.features = {}
    leaf._persisted_spawn_mandate = leaf_mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("LiveDepthRoot", root)
    admission, owns = await manager._admit_agent_operation(
        "LiveDepthLeaf",
        kind="spawn",
    )
    assert owns
    manager._do_spawn = AsyncMock(
        side_effect=AssertionError("live leaf bypassed its depth ceiling")
    )
    try:
        manager._prepare_agent_authority("LiveDepthLeaf", leaf)

        with pytest.raises(ValueError, match="max child depth"):
            await manager.spawn_agent(
                "ForbiddenLiveGrandchild",
                leaf,
                SpawnMandate(parent_did=leaf.agent_id),
            )
    finally:
        manager._withdraw_initialized_agent("LiveDepthLeaf", leaf)
        await manager._release_agent_operation(admission)

    manager._do_spawn.assert_not_awaited()
    assert admission.provisional_spawn_authority is False
    assert manager.get_mandate("LiveDepthLeaf") is None


def test_early_preflight_can_verify_private_batch_parent_without_granting_control():
    parent_did = "did:pkh:eip155:1:0xPrivateBatchParent"
    child_did = "did:pkh:eip155:1:0xPrivateBatchChild"
    parent, mandate = _signed_restored_mandate(parent_did, child_did)
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager()
    manager._initializing_agents["Parent"] = parent

    manager._verify_agent_authority("Child", child)

    assert manager.get_children(parent_did) == []
    assert manager.get_mandate("Child") is None
    with pytest.raises(
        PersistedSpawnParentUnavailableError,
        match="before its parent authority is loaded",
    ):
        manager._prepare_agent_authority("Child", child)

    manager._register_agent("Parent", parent)
    manager._prepare_agent_authority("Child", child)
    assert manager.get_children(parent_did) == ["Child"]


def test_cold_restore_rejects_child_beyond_parent_remaining_depth():
    root_did = "did:pkh:eip155:1:0xDepthRestoreRoot"
    parent_did = "did:pkh:eip155:1:0xDepthRestoreParent"
    child_did = "did:pkh:eip155:1:0xDepthRestoreChild"
    root, parent_mandate = _signed_restored_mandate(
        root_did,
        parent_did,
        max_child_depth=0,
    )
    parent, child_mandate = _signed_restored_mandate(
        parent_did,
        child_did,
        max_child_depth=0,
    )
    parent._persisted_spawn_mandate = parent_mandate
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = child_mandate
    manager = AgentManager()
    manager._register_agent("DepthRestoreRoot", root)
    manager._prepare_agent_authority("DepthRestoreParent", parent)
    manager._register_agent("DepthRestoreParent", parent)

    with pytest.raises(RuntimeError, match="child-depth authority"):
        manager._prepare_agent_authority("DepthRestoreChild", child)

    assert manager.get_mandate("DepthRestoreChild") is None


def test_initializing_signed_edges_reject_batch_cycle_before_projection():
    a_did = "did:pkh:eip155:1:0xBatchCycleA"
    b_did = "did:pkh:eip155:1:0xBatchCycleB"
    a_key, _ = generate_secp256k1_keypair()
    b_key, _ = generate_secp256k1_keypair()
    a = _make_mock_agent(a_did)
    b = _make_mock_agent(b_did)
    a._private_key = a_key
    b._private_key = b_key
    a.identity = None
    b.identity = None
    a._persisted_spawn_mandate = sign_mandate(
        SpawnMandate(parent_did=b_did, child_did=a_did),
        b_key,
    )
    b._persisted_spawn_mandate = sign_mandate(
        SpawnMandate(parent_did=a_did, child_did=b_did),
        a_key,
    )
    manager = AgentManager()
    manager._initializing_agents.update({"CycleA": a, "CycleB": b})

    with pytest.raises(RuntimeError, match="contains a cycle"):
        manager._verify_agent_authority("CycleA", a)

    assert manager._child_mandates == {}


@pytest.mark.asyncio
async def test_live_provisional_authority_reaches_cap_arbitration(tmp_path):
    parent_did = "did:pkh:eip155:1:0xLiveCapParent"
    child_did = "did:pkh:eip155:1:0xLiveCapChild"
    parent, mandate = _signed_restored_mandate(parent_did, child_did)
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._max_spawned_agents = 1
    manager._register_agent("LiveCapParent", parent)
    manager._child_mandates["ColdWinner"] = SpawnMandate(
        parent_did="did:cold:parent",
        child_did="did:cold:winner",
    )
    admission, owns = await manager._admit_agent_operation(
        "LiveCapChild",
        kind="spawn",
    )
    assert owns
    admission.spawn_parent = parent
    admission.spawn_slot_active = True
    manager._pending_spawns = 1
    try:
        manager._prepare_agent_authority("LiveCapChild", child)

        assert admission.provisional_spawn_authority is True
        assert manager.get_mandate("LiveCapChild") is mandate
    finally:
        manager._withdraw_initialized_agent("LiveCapChild", child)
        manager._pending_spawns = 0
        admission.spawn_slot_active = False
        await manager._release_agent_operation(admission)


@pytest.mark.asyncio
async def test_cold_cap_owner_is_not_displaced_by_provisional_live_projection(
    tmp_path,
):
    """The restored reservation wins regardless of projection timing."""

    cold_parent, cold_mandate = _signed_restored_mandate(
        "did:test:cold-cap-parent",
        "did:test:cold-cap-child",
    )
    live_parent, live_mandate = _signed_restored_mandate(
        "did:test:live-cap-parent",
        "did:test:live-cap-child",
    )
    cold = _make_mock_agent(cold_mandate.child_did)
    cold._persisted_spawn_mandate = cold_mandate
    live = _make_mock_agent(live_mandate.child_did)
    live._persisted_spawn_mandate = live_mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._max_spawned_agents = 1
    manager._register_agent("ColdParent", cold_parent)
    manager._register_agent("LiveParent", live_parent)
    manager._verify_agent_authority("ColdChild", cold)
    admission, owns = await manager._admit_agent_operation(
        "LiveChild",
        kind="spawn",
    )
    assert owns
    admission.spawn_parent = live_parent
    admission.spawn_slot_active = True
    admission.spawn_slot_terminal = asyncio.get_running_loop().create_future()
    manager._pending_spawns = 1
    try:
        manager._prepare_agent_authority("LiveChild", live)
        manager._prepare_agent_authority("ColdChild", cold)

        assert admission.provisional_spawn_authority is True
        assert manager.get_mandate("ColdChild") is cold_mandate
        assert manager.get_mandate("LiveChild") is live_mandate
    finally:
        manager._withdraw_initialized_agent("LiveChild", live)
        manager._withdraw_initialized_agent("ColdChild", cold)
        manager._pending_spawns = 0
        admission.spawn_slot_active = False
        if admission.spawn_slot_terminal is not None:
            admission.spawn_slot_terminal.set_result(None)
        await manager._release_agent_operation(admission)


@pytest.mark.asyncio
async def test_restored_ttl_cannot_remove_child_before_load_commits(tmp_path):
    """TTL adoption is the last onboarding commit, never a concurrent reaper."""

    parent_did = "did:pkh:eip155:1:0xSlowParent"
    child_did = "did:pkh:eip155:1:0xSlowChild"
    parent, mandate = _signed_restored_mandate(
        parent_did,
        child_did,
        ttl_seconds=1,
        created_at=(datetime.now(timezone.utc) - timedelta(seconds=0.9)).isoformat(),
    )
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    child_dir = tmp_path / "slow-child"
    child_dir.mkdir()
    (child_dir / "kestrel_prime.db").touch()
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("SlowParent", parent)
    manager._initialize_agent = AsyncMock(return_value=child)
    onboarding_observations: list[object] = []

    async def slow_onboarding(_name, _agent):
        onboarding_observations.append(manager.get_agent("SlowChild"))
        try:
            await asyncio.sleep(1.0)
        finally:
            onboarding_observations.append(manager.get_agent("SlowChild"))

    manager._on_agent_registered = AsyncMock(side_effect=slow_onboarding)

    with (
        patch(
            "kestrel_sovereign.multi_agent.agent_manager.remaining_spawn_ttl_seconds",
            return_value=0.5,
        ),
        pytest.raises(RuntimeError, match="expired during onboarding"),
    ):
        await manager.load_agent(
            "SlowChild",
            LocalAgentConfig(data_dir=child_dir, port=8801),
        )

    await asyncio.sleep(0)
    assert onboarding_observations == [None, None]
    assert manager.get_agent("SlowChild") is None
    assert manager.get_children(parent_did) == []
    assert (child_dir / ".kestrel-spawn-retired").read_text().strip() == child_did


def test_spawn_retirement_marker_fsyncs_directory_entry_before_return(
    tmp_path,
):
    """The tombstone itself and its directory entry are both durable."""

    child_name = "DurableRetirement"
    child_did = "did:test:durable-retirement"
    child_dir = tmp_path / "agent_data" / child_name
    child_dir.mkdir(parents=True)
    (child_dir / "kestrel_prime.db").touch()
    manager = AgentManager(base_data_dir=tmp_path)
    fsynced_modes: list[int] = []
    real_fsync = os.fsync

    def record_fsync(descriptor):
        fsynced_modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    with patch(
        "kestrel_sovereign.multi_agent.agent_manager.os.fsync",
        side_effect=record_fsync,
    ):
        marker = manager.record_expired_spawn_retirement(
            child_name,
            expected_child_did=child_did,
        )

    assert marker == child_dir / ".kestrel-spawn-retired"
    assert any(stat.S_ISREG(mode) for mode in fsynced_modes)
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)


def test_spawn_retirement_refreshes_stale_marker_for_replacement_identity(
    tmp_path,
):
    """A reused data directory cannot resurrect the replacement after expiry."""

    child_name = "ReplacementRetirement"
    child_did = "did:test:replacement-retirement"
    child_dir = tmp_path / "agent_data" / child_name
    child_dir.mkdir(parents=True)
    (child_dir / "kestrel_prime.db").touch()
    marker = child_dir / ".kestrel-spawn-retired"
    marker.write_text("did:test:retired-prior-identity\n")
    manager = AgentManager(base_data_dir=tmp_path)

    with patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did_sync",
        return_value=child_did,
    ):
        recorded = manager.record_expired_spawn_retirement(
            child_name,
            expected_child_did=child_did,
        )

    assert recorded == marker
    assert marker.read_text() == f"{child_did}\n"


def test_spawn_retirement_preserves_stale_marker_when_database_did_differs(
    tmp_path,
):
    """A late expiry owner cannot tombstone a newer same-name identity."""

    child_name = "RacingRetirement"
    expected_did = "did:test:expired-racing-identity"
    child_dir = tmp_path / "agent_data" / child_name
    child_dir.mkdir(parents=True)
    (child_dir / "kestrel_prime.db").touch()
    marker = child_dir / ".kestrel-spawn-retired"
    original = "did:test:retired-prior-identity\n"
    marker.write_text(original)
    manager = AgentManager(base_data_dir=tmp_path)

    with (
        patch(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did_sync",
            return_value="did:test:newer-live-identity",
        ),
        pytest.raises(RuntimeError, match="current database identity"),
    ):
        manager.record_expired_spawn_retirement(
            child_name,
            expected_child_did=expected_did,
        )

    assert marker.read_text() == original


@pytest.mark.asyncio
async def test_restored_ttl_cancels_ready_hook_at_signed_deadline(tmp_path):
    """Wake-capable readiness cannot run past an ephemeral mandate expiry."""

    parent_did = "did:pkh:eip155:1:0xDeadlineParent"
    child_did = "did:pkh:eip155:1:0xDeadlineChild"
    parent, mandate = _signed_restored_mandate(
        parent_did,
        child_did,
        ttl_seconds=2,
        created_at=(datetime.now(timezone.utc) - timedelta(seconds=1.4)).isoformat(),
    )
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    child_dir = tmp_path / "deadline-child"
    child_dir.mkdir()
    (child_dir / "kestrel_prime.db").touch()
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("DeadlineParent", parent)
    manager._initialize_agent = AsyncMock(return_value=child)
    wake_effects: list[str] = []

    async def wake_after_expiry(_agent):
        await asyncio.sleep(1.0)
        wake_effects.append("dispatched")

    manager._run_hosted_agent_ready_hooks = AsyncMock(
        side_effect=wake_after_expiry
    )

    with (
        patch(
            "kestrel_sovereign.multi_agent.agent_manager.remaining_spawn_ttl_seconds",
            return_value=0.5,
        ),
        pytest.raises(RuntimeError, match="expired during agent readiness"),
    ):
        await manager.load_agent(
            "DeadlineChild",
            LocalAgentConfig(data_dir=child_dir, port=8801),
        )

    assert wake_effects == []
    assert manager.get_agent("DeadlineChild") is None
    assert manager.get_children(parent_did) == []
    assert (child_dir / ".kestrel-spawn-retired").read_text().strip() == child_did


@pytest.mark.asyncio
async def test_final_publication_verification_cannot_finish_after_deadline(
    monkeypatch, tmp_path
):
    manager = AgentManager(base_data_dir=tmp_path)
    child = _make_mock_agent("did:test:final-publication-deadline")
    admission, owns = await manager._admit_agent_operation(
        "FinalDeadlineChild", kind="spawn"
    )
    assert owns
    manager._register_agent("FinalDeadlineChild", child)
    admission.published = True
    manager._run_hosted_agent_ready_hooks = AsyncMock()
    manager._rollback_published_agent_after_readiness_failure = AsyncMock(
        return_value=False
    )
    from kestrel_sovereign.multi_agent import agent_manager as manager_module

    original_join = manager_module.await_lifecycle_task_completion

    async def _delay_final_publication_join(task):
        if task.get_name().startswith("agent_readiness_publication_check:"):
            await asyncio.sleep(0.03)
        return await original_join(task)

    monkeypatch.setattr(
        manager_module,
        "await_lifecycle_task_completion",
        _delay_final_publication_join,
    )
    deadline = asyncio.get_running_loop().time() + 0.01
    try:
        with pytest.raises(RuntimeError, match="expired"):
            await manager._finish_published_agent_readiness(
                "FinalDeadlineChild",
                child,
                admission,
                deadline=deadline,
                failure_description="final publication failed",
            )
    finally:
        manager._withdraw_initialized_agent("FinalDeadlineChild", child)
        await manager._release_agent_operation(admission)


@pytest.mark.asyncio
async def test_expired_admission_fences_route_before_joining_stubborn_work(
    tmp_path,
):
    """A ready hook that swallows cancellation cannot extend routing authority."""

    manager = AgentManager(base_data_dir=tmp_path)
    child = _make_mock_agent("did:test:stubborn-expired-child")
    manager._register_agent("StubbornChild", child)
    release = asyncio.Event()
    cancellation_seen = asyncio.Event()

    async def stubborn_work():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()

    deadline = asyncio.get_running_loop().time() + 0.02
    expired = asyncio.create_task(
        manager._await_before_spawn_admission_deadline(
            stubborn_work,
            deadline=deadline,
            phase="stubborn readiness",
            on_expiry=lambda: manager._fence_expired_spawn_route(
                "StubbornChild", child
            ),
        )
    )
    await asyncio.wait_for(cancellation_seen.wait(), timeout=0.1)
    assert manager.get_agent("StubbornChild") is None
    assert "StubbornChild" not in manager.list_agents()
    assert len(manager._expired_admission_operations) == 1
    assert expired.done() is False

    release.set()
    with pytest.raises(RuntimeError, match="expired during stubborn readiness"):
        await asyncio.wait_for(expired, timeout=0.2)
    assert manager._expired_admission_operations == set()
    manager._withdraw_initialized_agent("StubbornChild", child)


@pytest.mark.asyncio
async def test_expired_batch_candidate_has_independent_private_cleanup(tmp_path):
    """A completed initializer cannot leave active services waiting on peers."""

    manager = AgentManager(base_data_dir=tmp_path)
    child = _make_mock_agent("did:test:expired-batch-candidate")
    manager._initializing_agents["ExpiredBatch"] = child
    boot_owner = asyncio.create_task(asyncio.sleep(0))
    await boot_owner
    child._host_authority_active_boot_task = boot_owner

    manager._expire_host_authority_candidate("ExpiredBatch", child)
    cleanup = child._host_authority_expiry_cleanup_task
    await asyncio.wait_for(asyncio.shield(cleanup), timeout=0.2)

    assert manager.get_agent("ExpiredBatch") is None
    assert "ExpiredBatch" not in manager._initializing_agents
    child.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_expired_batch_initializer_does_not_cancel_healthy_sibling(tmp_path):
    """One finite child expiry is an isolated startup failure, not batch cancel."""

    manager = AgentManager(base_data_dir=tmp_path)
    expiring = _make_mock_agent("did:test:expiring-batch-child")
    healthy = _make_mock_agent("did:test:healthy-batch-sibling")
    configs = {
        "ExpiringBatchChild": LocalAgentConfig(
            data_dir=tmp_path / "expiring-batch-child",
            port=8801,
        ),
        "HealthyBatchSibling": LocalAgentConfig(
            data_dir=tmp_path / "healthy-batch-sibling",
            port=8802,
        ),
    }

    async def initialize(name, _config):
        candidate = expiring if name == "ExpiringBatchChild" else healthy
        manager._initializing_agents[name] = candidate
        if candidate is healthy:
            return candidate
        candidate._host_authority_boot_expired = True
        manager._expire_host_authority_candidate(name, candidate)
        try:
            # Deliver cancellation from the expiry watchdog to its exact boot
            # owner, as the real initializer would at its next await.
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            await manager._discard_unpublished_initialized_agent(name, candidate)
            raise
        raise AssertionError("expiry did not cancel its candidate initializer")

    manager._initialize_agent = initialize

    batch = asyncio.create_task(
        manager.load_from_config(
            MultiAgentConfig(agents=configs),
            restart_roster_reconciled=True,
        )
    )
    loaded = await asyncio.wait_for(batch, timeout=1.0)

    assert loaded == 1
    assert manager.get_agent("HealthyBatchSibling") is healthy
    assert manager.get_agent("ExpiringBatchChild") is None
    assert len(manager.init_failures) == 1
    failed_name, failure = manager.init_failures[0]
    assert failed_name == "ExpiringBatchChild"
    assert isinstance(failure, PersistedSpawnMandateExpiredError)
    expiring.shutdown.assert_awaited_once()
    healthy.shutdown.assert_not_awaited()


@pytest.mark.asyncio
async def test_expired_published_candidate_keeps_one_readiness_rollback_owner(
    tmp_path,
):
    """The watchdog fences publication but cannot race receipt-first rollback."""

    manager = AgentManager(base_data_dir=tmp_path)
    child = _make_mock_agent("did:test:expired-published-candidate")
    admission, owns = await manager._admit_agent_operation(
        "ExpiredPublished", kind="spawn"
    )
    assert owns
    admission.child = child
    admission.published = True
    manager._register_agent("ExpiredPublished", child)
    completed_boot = asyncio.create_task(asyncio.sleep(0))
    await completed_boot
    child._host_authority_active_boot_task = completed_boot
    manager._discard_unpublished_initialized_agent = AsyncMock()

    try:
        manager._expire_host_authority_candidate("ExpiredPublished", child)
        await asyncio.sleep(0)

        assert manager._spawn_route_is_fenced("ExpiredPublished", child)
        assert vars(child).get("_host_authority_expiry_cleanup_task") is None
        manager._discard_unpublished_initialized_agent.assert_not_awaited()
    finally:
        manager._withdraw_initialized_agent("ExpiredPublished", child)
        await manager._release_agent_operation(admission)


@pytest.mark.asyncio
async def test_expired_private_candidate_cancels_admission_owner_and_shuts_down(
    tmp_path,
):
    """A post-boot child cannot run forever while awaiting parent publication."""

    parent_did = "did:test:stalled-publication-parent"
    child_did = "did:test:stalled-publication-child"
    parent, mandate = _signed_restored_mandate(
        parent_did,
        child_did,
        ttl_seconds=1,
    )
    parent._agent_manager_publication_event = asyncio.Event()
    parent._agent_manager_published = False
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    initialized = asyncio.Event()
    manager = AgentManager(base_data_dir=tmp_path)
    manager._initializing_agents["StalledParent"] = parent

    async def initialize(name, _config, **_kwargs):
        manager._initializing_agents[name] = child
        child._host_authority_expiry_callback = (
            lambda candidate: manager._expire_host_authority_candidate(
                name, candidate
            )
        )
        arm_host_authority_deadline(child, mandate)
        # Model KestrelAgent.initialize() returning successfully: the original
        # boot owner is finished, while the manager admission still owns the
        # initialized private runtime through publication or rollback.
        vars(child).pop("_host_authority_active_boot_task", None)
        initialized.set()
        return child

    manager._initialize_agent = initialize
    with patch(
        "kestrel_sovereign.kestrel_agent.remaining_spawn_ttl_seconds",
        return_value=0.03,
    ):
        load = asyncio.create_task(
            manager.load_agent(
                "StalledChild",
                LocalAgentConfig(data_dir="unused", port=8801),
            )
        )
        await asyncio.wait_for(initialized.wait(), timeout=0.2)
        try:
            await asyncio.sleep(0.08)
            assert load.done(), "signed expiry must cancel the admission owner"
            with pytest.raises(asyncio.CancelledError):
                await load
        finally:
            if not load.done():
                load.cancel()
                await asyncio.gather(load, return_exceptions=True)

    assert manager.get_agent("StalledChild") is None
    assert "StalledChild" not in manager._initializing_agents
    assert manager._agent_operations == {}
    child.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_already_expired_host_deadline_does_not_leave_latent_cancellation():
    """The synchronous expiry exception is the current task's sole outcome."""

    agent = SimpleNamespace()
    mandate = SpawnMandate(
        parent_did="did:test:expired-deadline-parent",
        child_did="did:test:expired-deadline-child",
        ttl_seconds=1,
        created_at=(datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat(),
        parent_signature="signed-expired-deadline",
    )

    async def arm_then_continue() -> tuple[int, bool]:
        owner = asyncio.current_task()
        assert owner is not None
        with pytest.raises(PersistedSpawnMandateExpiredError):
            arm_host_authority_deadline(agent, mandate)
        cancelling = owner.cancelling()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            return cancelling, False
        return cancelling, True

    assert await asyncio.create_task(arm_then_continue()) == (0, True)


@pytest.mark.asyncio
async def test_unconfirmed_private_shutdown_retains_runtime_namespace(tmp_path):
    manager = AgentManager(base_data_dir=tmp_path)
    child = _make_mock_agent("did:test:unconfirmed-private-shutdown")
    admission, owns = await manager._admit_agent_operation(
        "UnconfirmedPrivateChild", kind="spawn"
    )
    assert owns
    admission.unpublished_cleanup_deferred_to_spawn = True
    manager._discard_unpublished_initialized_agent = AsyncMock(
        side_effect=RuntimeError("shutdown completion unconfirmed")
    )
    manager._offboard_agent_runtime_namespace = AsyncMock(
        return_value=(False, None)
    )
    try:
        with pytest.raises(BaseException, match="shutdown completion unconfirmed"):
            await manager._rollback_uncommitted_spawn_runtime(admission, child)

        manager._offboard_agent_runtime_namespace.assert_not_awaited()
        assert admission.unpublished_cleanup_deferred_to_spawn is True
    finally:
        await manager._release_agent_operation(admission)


@pytest.mark.asyncio
async def test_rejected_private_spawn_releases_budget_and_retires_storage(
    tmp_path,
):
    """Private rollback owns both the delegated hold and durable discovery."""

    name = "RejectedPrivateChild"
    child_did = "did:test:rejected-private-child"
    agent_data = tmp_path / "agent_data"
    child_dir = agent_data / name
    child_dir.mkdir(parents=True)
    (child_dir / "kestrel_prime.db").touch()

    manager = AgentManager(base_data_dir=tmp_path)
    child = _make_mock_agent(child_did)
    admission, owns = await manager._admit_agent_operation(name, kind="spawn")
    assert owns
    admission.unpublished_cleanup_deferred_to_spawn = True
    manager._child_budgets[name] = (object(), object())
    manager._discard_unpublished_initialized_agent = AsyncMock()
    manager._offboard_agent_runtime_namespace = AsyncMock(
        return_value=(False, None)
    )

    async def release_budget(child_name: str) -> bool:
        assert child_name == name
        manager._child_budgets.pop(child_name)
        return False

    manager._release_child_budget_cancellation_safe = release_budget
    try:
        await manager._rollback_uncommitted_spawn_runtime(admission, child)

        assert name not in manager._child_budgets
        assert name not in MultiAgentConfig.auto_discover(agent_data).agents
        assert admission.unpublished_cleanup_deferred_to_spawn is False
    finally:
        await manager._release_agent_operation(admission)


@pytest.mark.asyncio
async def test_rejected_hosted_spawn_is_retired_after_successful_removal(
    tmp_path,
):
    """Successful runtime removal cannot bypass durable retirement."""

    name = "RejectedHostedChild"
    child_did = "did:test:rejected-hosted-child"
    agent_data = tmp_path / "agent_data"
    child_dir = agent_data / name
    child_dir.mkdir(parents=True)
    (child_dir / "kestrel_prime.db").touch()

    manager = AgentManager(base_data_dir=tmp_path)
    child = _make_mock_agent(child_did)
    admission, owns = await manager._admit_agent_operation(name, kind="spawn")
    assert owns
    manager.remove_agent = AsyncMock(return_value=True)
    manager.terminate_children = AsyncMock()
    try:
        await manager._rollback_uncommitted_spawn_runtime(admission, child)

        assert name not in MultiAgentConfig.auto_discover(agent_data).agents
    finally:
        await manager._release_agent_operation(admission)


@pytest.mark.asyncio
async def test_rejected_storage_spawn_is_retired_before_rollback_completes(
    tmp_path,
):
    """A failed signed spawn cannot reappear through config-less discovery."""

    name = "RejectedStorageChild"
    child_did = "did:test:rejected-storage-child"
    agent_data = tmp_path / "agent_data"
    child_dir = agent_data / name
    child_dir.mkdir(parents=True)
    (child_dir / "kestrel_prime.db").touch()
    assert name in MultiAgentConfig.auto_discover(agent_data).agents

    manager = AgentManager(base_data_dir=tmp_path)
    child = _make_mock_agent(child_did)
    admission, owns = await manager._admit_agent_operation(name, kind="spawn")
    assert owns
    manager.remove_agent = AsyncMock(
        side_effect=RuntimeOffboardingNotPerformedError(
            agent_name=name,
            agent_id=child_did,
            cleanup_state="not_hosted",
        )
    )
    manager.terminate_children = AsyncMock()
    try:
        await manager._rollback_uncommitted_spawn_runtime(admission, child)

        assert name not in MultiAgentConfig.auto_discover(agent_data).agents
    finally:
        await manager._release_agent_operation(admission)


@pytest.mark.asyncio
async def test_signed_parent_lookup_ignores_unrelated_unresolved_admission(
    tmp_path,
):
    """A published parent is sufficient despite an unrelated slow anchor read."""

    manager = AgentManager(base_data_dir=tmp_path)
    parent_did = "did:test:already-published-parent"
    manager._register_agent("PublishedParent", _make_mock_agent(parent_did))
    unrelated, owns = await manager._admit_agent_operation(
        "SlowUnrelated",
        kind="load",
    )
    assert owns
    try:
        await asyncio.wait_for(
            manager._await_admitted_parent_candidate(
                "SignedChild",
                parent_did,
            ),
            timeout=0.05,
        )
    finally:
        assert unrelated.agent_id_resolution_event is not None
        unrelated.agent_id_resolution_event.set()
        await manager._release_agent_operation(unrelated)


@pytest.mark.asyncio
async def test_initial_spawn_cap_does_not_double_count_provisional_projection(
    tmp_path,
):
    manager = AgentManager(base_data_dir=tmp_path)
    manager._max_spawned_agents = 2
    parent = _make_mock_agent("did:test:spawn-cap-parent")
    manager._register_agent("SpawnCapParent", parent)

    provisional, owns = await manager._admit_agent_operation(
        "ProvisionalChild", kind="spawn"
    )
    assert owns
    provisional.spawn_slot_active = True
    provisional.provisional_spawn_authority = True
    manager._pending_spawns = 1
    manager._child_mandates["ProvisionalChild"] = SpawnMandate(
        parent_did=parent.agent_id,
        child_did="did:test:provisional-child",
    )

    second, owns = await manager._admit_agent_operation(
        "SecondChild", kind="spawn"
    )
    assert owns
    manager._do_spawn = AsyncMock(
        side_effect=RuntimeError("second spawn reached implementation")
    )
    try:
        with pytest.raises(RuntimeError, match="reached implementation"):
            await manager._run_admitted_spawn(
                "SecondChild",
                parent,
                SpawnMandate(parent_did=parent.agent_id),
                second,
            )
        manager._do_spawn.assert_awaited_once()
    finally:
        manager._child_mandates.pop("ProvisionalChild", None)
        manager._pending_spawns = 0
        provisional.spawn_slot_active = False
        provisional.provisional_spawn_authority = False
        await manager._release_agent_operation(provisional)
        await manager._release_agent_operation(second)


@pytest.mark.asyncio
async def test_live_spawn_recomputes_ready_deadline_after_receipt_is_signed(
    tmp_path,
):
    """The final signed timestamp, not the unsigned proposal, bounds readiness."""

    parent = _make_mock_agent("did:test:fresh-deadline-parent")
    private_key, _ = generate_secp256k1_keypair()
    parent._private_key = private_key
    parent.identity = None
    child = _make_mock_agent("did:test:fresh-deadline-child")
    child._persisted_spawn_mandate = None
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("FreshDeadlineParent", parent)
    manager._initialize_agent = AsyncMock(return_value=child)
    admission, owns = await manager._admit_agent_operation(
        "FreshDeadlineChild",
        kind="spawn",
    )
    assert owns
    wake_effects: list[str] = []

    async def sign_receipt(candidate):
        mandate = SpawnMandate(
            parent_did=parent.agent_id,
            child_did=candidate.agent_id,
            ttl_seconds=1,
        )
        candidate._persisted_spawn_mandate = sign_mandate(mandate, private_key)

    async def wake_after_expiry(_candidate):
        await asyncio.sleep(1.0)
        wake_effects.append("dispatched")

    admission.before_publish = sign_receipt
    manager._run_hosted_agent_ready_hooks = AsyncMock(
        side_effect=wake_after_expiry
    )
    try:
        with (
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.remaining_spawn_ttl_seconds",
                return_value=0.5,
            ),
            pytest.raises(RuntimeError, match="expired during agent readiness"),
        ):
            await manager.load_agent(
                "FreshDeadlineChild",
                LocalAgentConfig(data_dir="unused", port=8801),
            )
    finally:
        await manager._release_agent_operation(admission)

    assert wake_effects == []
    assert manager.get_agent("FreshDeadlineChild") is None
    assert manager._prepared_agent_names == {}


@pytest.mark.asyncio
async def test_delete_joins_active_spawn_before_closing_child_storage(tmp_path):
    """DELETE cannot close the graph still owned by spawn receipt rollback."""

    manager = AgentManager(base_data_dir=tmp_path)
    child = _make_mock_agent("did:test:delete-during-spawn")
    graph = SimpleNamespace(closed=False)
    child._raw_storage = SimpleNamespace(graph=graph)

    async def shutdown():
        graph.closed = True

    child.shutdown = AsyncMock(side_effect=shutdown)
    spawn_started = asyncio.Event()
    release_spawn = asyncio.Event()

    async def active_spawn_owner():
        admission, owns = await manager._admit_agent_operation(
            "DeleteDuringSpawn", kind="spawn"
        )
        assert owns
        admission.spawn_task = asyncio.current_task()
        manager._agents["DeleteDuringSpawn"] = child
        manager._agent_names[child.agent_id] = "DeleteDuringSpawn"
        spawn_started.set()
        try:
            await release_spawn.wait()
        finally:
            await manager._release_agent_operation(admission)

    spawn_task = asyncio.create_task(active_spawn_owner())
    await asyncio.wait_for(spawn_started.wait(), timeout=1)
    delete_task = asyncio.create_task(manager.remove_agent("DeleteDuringSpawn"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert not graph.closed
    child.shutdown.assert_not_awaited()

    release_spawn.set()
    await asyncio.wait_for(spawn_task, timeout=1)
    assert await asyncio.wait_for(delete_task, timeout=1) is True
    assert graph.closed


@pytest.mark.asyncio
async def test_spawn_join_tracks_operation_not_callers_later_work(tmp_path):
    """A terminal drain stops waiting when the admitted spawn itself settles."""

    manager = AgentManager(base_data_dir=tmp_path)
    parent = _make_mock_agent("did:test:operation-owner-parent")
    parent.features = {}
    child = _make_mock_agent("did:test:operation-owner-child")
    spawn_entered = asyncio.Event()
    release_spawn = asyncio.Event()
    spawn_returned = asyncio.Event()
    release_caller = asyncio.Event()

    async def controlled_spawn(_name, _parent, _mandate, _admission):
        assert _admission.owner_task is asyncio.current_task()
        assert _admission.spawn_task is asyncio.current_task()
        spawn_entered.set()
        await release_spawn.wait()
        return child

    manager._do_spawn = AsyncMock(side_effect=controlled_spawn)

    async def caller_with_later_work():
        result = await manager.spawn_agent(
            "OperationOwnedChild",
            parent,
            SpawnMandate(parent_did=parent.agent_id),
        )
        assert result is child
        spawn_returned.set()
        await release_caller.wait()

    caller = asyncio.create_task(caller_with_later_work())
    await asyncio.wait_for(spawn_entered.wait(), timeout=1)
    join = asyncio.create_task(manager._join_admitted_spawn_operations())
    await asyncio.sleep(0)
    release_spawn.set()
    await asyncio.wait_for(spawn_returned.wait(), timeout=1)
    await asyncio.sleep(0)

    try:
        assert join.done(), "spawn join followed the enclosing caller task"
        assert join.result() == (False, [])
        assert not caller.done()
    finally:
        release_caller.set()
        await asyncio.wait_for(caller, timeout=1)
        await asyncio.wait_for(join, timeout=1)


@pytest.mark.asyncio
async def test_spawn_revokes_ambiguous_receipt_before_child_storage_closes(tmp_path):
    """A post-commit write error must not close the only revocation handle."""

    events: list[str] = []

    class PostCommitErrorGraph:
        def __init__(self) -> None:
            self.closed = False
            self.write_count = 0

        async def add_trusted_cross_agent_edge(
            self, _source, _target, _label, *, properties
        ) -> None:
            assert not self.closed
            self.write_count += 1
            events.append(
                "signed" if properties.get("parent_signature") else "revoked"
            )
            if self.write_count == 1:
                # Model a database commit followed by a transport failure.
                raise RuntimeError("post-commit transport failure")

    graph = PostCommitErrorGraph()
    child = _make_mock_agent("did:test:ambiguous-receipt-child")
    child._raw_storage = SimpleNamespace(graph=graph)

    async def close_child() -> None:
        graph.closed = True
        events.append("shutdown")

    child.shutdown = AsyncMock(side_effect=close_child)
    parent = _make_mock_agent("did:test:ambiguous-receipt-parent")
    parent._private_key, _ = generate_secp256k1_keypair()
    parent.identity = None
    parent.features = {}
    manager = AgentManager(base_data_dir=tmp_path)
    manager._initialize_agent = AsyncMock(return_value=child)

    async def create_through_real_load(name, **kwargs):
        return await _load_spawn_after_mocked_inception(manager, name, kwargs)

    manager.create_agent = AsyncMock(side_effect=create_through_real_load)

    with pytest.raises(RuntimeError, match="post-commit transport failure"):
        await manager.spawn_agent(
            "AmbiguousReceiptChild",
            parent,
            SpawnMandate(parent_did=parent.agent_id),
        )

    assert events == ["signed", "revoked", "shutdown"]
    assert graph.closed is True
    assert manager.get_agent("AmbiguousReceiptChild") is None
    witness = manager._spawn_authority_registry.get(child.agent_id)
    assert witness is not None and witness.state == "retired"


@pytest.mark.asyncio
async def test_failed_published_spawn_retains_cleanup_when_receipt_revocation_fails(
    tmp_path,
):
    """A signed published child keeps an owner until revocation and shutdown."""

    events: list[str] = []

    class TransientRevocationFailureGraph:
        def __init__(self) -> None:
            self.write_count = 0

        async def add_trusted_cross_agent_edge(
            self, _source, _target, _label, *, properties
        ) -> None:
            self.write_count += 1
            if properties.get("parent_signature"):
                events.append("signed")
                return
            if self.write_count == 2:
                events.append("revocation-failed")
                raise RuntimeError("revocation unavailable")
            events.append("revoked")

    graph = TransientRevocationFailureGraph()
    child = _make_mock_agent("did:test:published-revocation-child")
    child._raw_storage = SimpleNamespace(graph=graph)
    parent = _make_mock_agent("did:test:published-revocation-parent")
    parent._private_key, _ = generate_secp256k1_keypair()
    parent.identity = None
    parent.features = {}
    manager = AgentManager(base_data_dir=tmp_path)

    async def create_child(name, **kwargs):
        await _persist_and_publish_spawn_test_child(manager, name, child, kwargs)
        return child

    async def remove_child(name, **_kwargs):
        events.append("shutdown")
        manager._agents.pop(name, None)
        manager._agent_names.pop(child.agent_id, None)
        return True

    manager.create_agent = AsyncMock(side_effect=create_child)
    manager.remove_agent = AsyncMock(side_effect=remove_child)
    manager._apply_delegated_budget = AsyncMock(
        side_effect=RuntimeError("budget provider failed")
    )

    with pytest.raises(ExceptionGroup, match="owned rollback failed"):
        await manager.spawn_agent(
            "PublishedRevocationChild",
            parent,
            SpawnMandate(parent_did=parent.agent_id),
        )

    quarantined = manager.quarantined_shutdowns()
    assert len(quarantined) == 1
    await asyncio.wait_for(manager.drain_quarantined_shutdowns(), timeout=1.0)

    assert events == ["signed", "revocation-failed", "revoked", "shutdown"]
    assert manager.get_agent("PublishedRevocationChild") is None
    witness = manager._spawn_authority_registry.get(child.agent_id)
    assert witness is not None and witness.state == "retired"


@pytest.mark.asyncio
async def test_failed_spawn_quarantine_retains_cap_slot_until_child_is_removed(
    tmp_path,
):
    """A quarantined, mandate-less child still consumes a fleet-cap slot."""

    allow_revocation = asyncio.Event()

    class BlockedRevocationGraph:
        def __init__(self) -> None:
            self.write_count = 0

        async def add_trusted_cross_agent_edge(
            self, _source, _target, _label, *, properties
        ) -> None:
            self.write_count += 1
            if properties.get("parent_signature"):
                return
            if self.write_count == 2:
                raise RuntimeError("revocation unavailable")
            await allow_revocation.wait()

    graph = BlockedRevocationGraph()
    child = _make_mock_agent("did:test:quarantined-cap-child")
    child._raw_storage = SimpleNamespace(graph=graph)
    parent = _make_mock_agent("did:test:quarantined-cap-parent")
    parent._private_key, _ = generate_secp256k1_keypair()
    parent.identity = None
    parent.features = {}
    manager = AgentManager(base_data_dir=tmp_path)
    manager._max_spawned_agents = 1

    async def create_child(name, **kwargs):
        await _persist_and_publish_spawn_test_child(manager, name, child, kwargs)
        return child

    async def remove_child(name, **_kwargs):
        manager._agents.pop(name, None)
        manager._agent_names.pop(child.agent_id, None)
        return True

    manager.create_agent = AsyncMock(side_effect=create_child)
    manager.remove_agent = AsyncMock(side_effect=remove_child)
    manager._apply_delegated_budget = AsyncMock(
        side_effect=RuntimeError("budget provider failed")
    )

    with pytest.raises(ExceptionGroup, match="owned rollback failed"):
        await manager.spawn_agent(
            "QuarantinedCapChild",
            parent,
            SpawnMandate(parent_did=parent.agent_id),
        )

    assert manager._pending_spawns == 1
    with pytest.raises(ValueError, match="spawned-agent cap"):
        await manager.spawn_agent(
            "MustWaitForQuarantine",
            parent,
            SpawnMandate(parent_did=parent.agent_id),
        )

    allow_revocation.set()
    await asyncio.wait_for(manager.drain_quarantined_shutdowns(), timeout=1.0)
    assert manager._pending_spawns == 0


@pytest.mark.asyncio
async def test_failed_spawn_cleanup_waits_out_terminal_handoff_seal(tmp_path):
    """A terminal drain seal cannot orphan a newly-created cleanup task."""

    graph = SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
    child = _make_mock_agent("did:test:sealed-handoff-child")
    child._raw_storage = SimpleNamespace(graph=graph)
    manager = AgentManager(base_data_dir=tmp_path)
    admission = AgentOperationAdmission(
        name="SealedHandoffChild",
        canonical_name="sealedhandoffchild",
        kind="spawn",
        registration_epoch=0,
        owner_task=asyncio.current_task(),
        child=child,
        spawn_slot_active=True,
        spawn_receipt_graph=graph,
        spawn_receipt_source_id=child.agent_id,
        spawn_receipt_target_id="did:test:sealed-handoff-parent",
        spawn_receipt_unsigned_properties={"parent_signature": None},
    )
    manager._pending_spawns = 1
    manager._quarantined_shutdown_handoffs_sealed = True
    manager._quarantined_shutdown_handoffs_open.clear()
    await manager._quarantined_shutdown_drain_lock.acquire()
    manager._rollback_uncommitted_spawn_runtime = AsyncMock(return_value=False)

    handoff = asyncio.create_task(
        manager._handoff_failed_spawn_cleanup(admission, child)
    )
    await asyncio.sleep(0)
    assert not handoff.done()
    assert manager._quarantined_shutdown_reapers == {}

    manager._quarantined_shutdown_handoffs_sealed = False
    manager._quarantined_shutdown_handoffs_open.set()
    manager._quarantined_shutdown_drain_lock.release()
    await asyncio.wait_for(handoff, timeout=1.0)
    await asyncio.wait_for(manager.drain_quarantined_shutdowns(), timeout=1.0)

    assert manager._rollback_uncommitted_spawn_runtime.await_count == 1
    assert manager._pending_spawns == 0


@pytest.mark.asyncio
async def test_runtime_rollback_failure_retains_cleanup_and_withdraws_authority(
    tmp_path,
):
    """A routable rollback failure keeps an owner but loses signed power."""

    parent_did = "did:pkh:eip155:1:0xRollbackParent"
    child_did = "did:pkh:eip155:1:0xRollbackChild"
    _parent, mandate = _signed_restored_mandate(parent_did, child_did)
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    graph = SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
    manager = AgentManager(base_data_dir=tmp_path)
    name = "RollbackChild"
    canonical_name = manager._canonical_agent_name(name)
    admission = AgentOperationAdmission(
        name=name,
        canonical_name=canonical_name,
        kind="spawn",
        registration_epoch=0,
        owner_task=asyncio.current_task(),
        child=child,
        provisional_spawn_authority=True,
        spawn_slot_active=True,
        spawn_receipt_graph=graph,
        spawn_receipt_source_id=child_did,
        spawn_receipt_target_id=parent_did,
        spawn_receipt_unsigned_properties={"parent_signature": None},
    )
    manager._agent_operations[canonical_name] = admission
    manager._agents[name] = child
    manager._agent_names[child_did] = name
    manager._child_mandates[name] = mandate
    manager._parent_children[parent_did] = [name]
    manager._pending_spawns = 1
    allow_retained_cleanup = asyncio.Event()
    remove_calls = 0

    async def fail_once_then_remove(_name, **_kwargs):
        nonlocal remove_calls
        remove_calls += 1
        if remove_calls == 1:
            return False
        await allow_retained_cleanup.wait()
        manager._agents.pop(name, None)
        manager._agent_names.pop(child_did, None)
        return True

    manager.remove_agent = AsyncMock(side_effect=fail_once_then_remove)
    manager.terminate_children = AsyncMock()

    with pytest.raises(RuntimeError, match="live routable child"):
        await manager._rollback_uncommitted_spawn(admission, child)

    retained_snapshot = {
        "reaper": bool(manager._quarantined_shutdown_reapers),
        "signature": child._persisted_spawn_mandate.parent_signature,
        "mandate_present": name in manager._child_mandates,
        "non_governing": child_did in manager._non_governing_spawn_lineage,
        "pending_spawns": manager._pending_spawns,
        "slot_active": admission.spawn_slot_active,
        "cap_slots": manager._spawn_cap_slots_in_use(),
    }
    try:
        allow_retained_cleanup.set()
        await asyncio.wait_for(
            manager.drain_quarantined_shutdowns(),
            timeout=1.0,
        )
    finally:
        allow_retained_cleanup.set()

    assert retained_snapshot == {
        "reaper": True,
        "signature": None,
        "mandate_present": False,
        "non_governing": True,
        "pending_spawns": 1,
        "slot_active": False,
        "cap_slots": 1,
    }
    assert manager.get_agent(name) is None
    assert manager._pending_spawns == 0


@pytest.mark.asyncio
async def test_failed_spawn_persists_retirement_before_receipt_downgrade(tmp_path):
    """A crash after receipt downgrade must leave a durable restart denial."""

    child_name = "CrashOrderedRollbackChild"
    child_did = "did:test:crash-ordered-rollback-child"
    parent_did = "did:test:crash-ordered-rollback-parent"
    _parent, mandate = _signed_restored_mandate(parent_did, child_did)
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    async def assert_retiring_before_receipt_write(*_args, **_kwargs):
        witness = manager._spawn_authority_registry.get(child_did)
        assert witness is not None and witness.state == "retiring"

    graph = SimpleNamespace(
        add_trusted_cross_agent_edge=AsyncMock(
            side_effect=assert_retiring_before_receipt_write
        )
    )
    config = LocalAgentConfig(
        data_dir=Path("agent_data") / child_name,
        port=8802,
    )
    config_path = tmp_path / "multi_agent.toml"
    MultiAgentConfig(agents={child_name: config}).save(config_path)
    manager = AgentManager(
        base_data_dir=tmp_path,
        startup_config_path=config_path,
    )
    manager._spawn_authority_registry.record_active(
        child_name=child_name,
        child_did=child_did,
        mandate=mandate,
        config=config,
    )
    admission = AgentOperationAdmission(
        name=child_name,
        canonical_name=child_name.casefold(),
        kind="spawn",
        registration_epoch=0,
        owner_task=asyncio.current_task(),
        child=child,
        spawn_receipt_graph=graph,
        spawn_receipt_source_id=child_did,
        spawn_receipt_target_id=parent_did,
        spawn_receipt_unsigned_properties={"parent_signature": None},
        spawn_authority_witness_did=child_did,
        spawn_authority_witness_mandate=mandate,
        spawn_startup_config=config,
        spawn_startup_config_path=config_path,
    )
    async def retire_runtime(_admission, _child):
        witness = manager._spawn_authority_registry.get(child_did)
        assert witness is not None and witness.state == "retiring"
        manager._spawn_authority_registry.retire(
            child_name=child_name,
            child_did=child_did,
        )
        return False

    manager._rollback_uncommitted_spawn_runtime = AsyncMock(
        side_effect=retire_runtime
    )
    manager._handoff_failed_spawn_cleanup = AsyncMock(return_value=False)

    with patch.object(
        manager._spawn_authority_registry,
        "withdraw_active",
        wraps=manager._spawn_authority_registry.withdraw_active,
    ) as withdraw_witness:
        assert await manager._rollback_uncommitted_spawn(admission, child) is False

    withdraw_witness.assert_not_called()
    assert MultiAgentConfig.from_file(config_path).agents == {}
    witness = manager._spawn_authority_registry.get(child_did)
    assert witness is not None and witness.state == "retired"
    assert admission.spawn_authority_witness_did is None
    assert admission.spawn_authority_witness_mandate is None
    manager._handoff_failed_spawn_cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_spawn_handoff_preserves_roster_witness_runtime_order(tmp_path):
    """The retained cleanup owner retries every crash-ordered rollback rail."""

    child = _make_mock_agent("did:test:retained-order-child")
    config = LocalAgentConfig(data_dir=Path("agent_data/Child"), port=8802)
    mandate = SpawnMandate(
        parent_did="did:test:retained-order-parent",
        child_did=child.agent_id,
        parent_signature="retained-order-signature",
    )
    manager = AgentManager(base_data_dir=tmp_path)
    admission = AgentOperationAdmission(
        name="RetainedOrderChild",
        canonical_name="retainedorderchild",
        kind="spawn",
        registration_epoch=0,
        owner_task=asyncio.current_task(),
        child=child,
        spawn_receipt_graph=object(),
        spawn_startup_config=config,
        spawn_startup_config_path=tmp_path / "multi_agent.toml",
        spawn_authority_witness_did=child.agent_id,
        spawn_authority_witness_mandate=mandate,
    )
    events = []

    async def receipt(_admission, _child):
        events.append("receipt")
        admission.spawn_receipt_graph = None
        return False

    async def roster(_admission):
        events.append("roster")
        admission.spawn_startup_config = None
        admission.spawn_startup_config_path = None
        return False

    async def witness(_admission):
        events.append("witness")
        admission.spawn_authority_witness_did = None
        admission.spawn_authority_witness_mandate = None
        return False

    async def runtime(_admission, _child):
        events.append("runtime")
        return False

    manager._downgrade_uncommitted_spawn_receipt = AsyncMock(side_effect=receipt)
    manager._withdraw_uncommitted_spawn_startup_registration = AsyncMock(
        side_effect=roster
    )
    manager._withdraw_uncommitted_spawn_authority_witness = AsyncMock(
        side_effect=witness
    )
    manager._rollback_uncommitted_spawn_runtime = AsyncMock(side_effect=runtime)

    assert await manager._handoff_failed_spawn_cleanup(admission, child) is False
    assert await manager.drain_quarantined_shutdowns() is False

    assert events == ["receipt", "roster", "runtime", "witness"]


@pytest.mark.asyncio
async def test_failed_spawn_cleanup_does_not_reacquire_owning_drain_lock(tmp_path):
    """Fleet shutdown can join a spawn while already owning the drain lock."""

    graph = SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
    child = _make_mock_agent("did:test:joined-spawn-child")
    child._raw_storage = SimpleNamespace(graph=graph)
    manager = AgentManager(base_data_dir=tmp_path)
    admission = AgentOperationAdmission(
        name="JoinedSpawnChild",
        canonical_name="joinedspawnchild",
        kind="spawn",
        registration_epoch=0,
        owner_task=asyncio.current_task(),
        child=child,
        spawn_slot_active=True,
        spawn_receipt_graph=graph,
        spawn_receipt_source_id=child.agent_id,
        spawn_receipt_target_id="did:test:joined-spawn-parent",
        spawn_receipt_unsigned_properties={"parent_signature": None},
    )
    manager._pending_spawns = 1
    manager._rollback_uncommitted_spawn_runtime = AsyncMock(return_value=False)
    await manager._quarantined_shutdown_drain_lock.acquire()
    handoff = asyncio.create_task(
        manager._handoff_failed_spawn_cleanup(admission, child)
    )
    try:
        done, _pending = await asyncio.wait({handoff}, timeout=0.05)
    finally:
        manager._quarantined_shutdown_drain_lock.release()
    if handoff not in done:
        await asyncio.wait_for(handoff, timeout=1.0)
    assert handoff in done
    assert handoff.result() is False

    await asyncio.wait_for(manager.drain_quarantined_shutdowns(), timeout=1.0)
    assert manager._pending_spawns == 0


@pytest.mark.asyncio
async def test_over_cap_rejected_spawn_keeps_slot_if_rollback_is_quarantined(
    tmp_path,
):
    """A restored child cannot open a cap gap around failed revocation."""

    allow_revocation = asyncio.Event()

    class BlockedRevocationGraph:
        def __init__(self) -> None:
            self.write_count = 0

        async def add_trusted_cross_agent_edge(
            self, _source, _target, _label, *, properties
        ) -> None:
            self.write_count += 1
            if properties.get("parent_signature"):
                return
            if self.write_count == 2:
                raise RuntimeError("revocation unavailable")
            await allow_revocation.wait()

    manager = AgentManager(base_data_dir=tmp_path)
    manager._max_spawned_agents = 1
    parent = _make_mock_agent("did:test:over-cap-quarantine-parent")
    parent._private_key, _ = generate_secp256k1_keypair()
    parent.identity = None
    parent.features = {}
    graph = BlockedRevocationGraph()
    child = _make_mock_agent("did:test:over-cap-quarantine-child")
    child._raw_storage = SimpleNamespace(graph=graph)

    async def create_after_cold_restore(name, **kwargs):
        await _persist_and_publish_spawn_test_child(manager, name, child, kwargs)
        manager._child_mandates["RestoredChild"] = SpawnMandate(
            parent_did="did:test:other-parent",
            child_did="did:test:restored-child",
        )
        return child

    async def remove_child(name, **_kwargs):
        manager._agents.pop(name, None)
        manager._agent_names.pop(child.agent_id, None)
        return True

    manager.create_agent = AsyncMock(side_effect=create_after_cold_restore)
    manager.remove_agent = AsyncMock(side_effect=remove_child)

    with pytest.raises(ExceptionGroup, match="owned rollback failed"):
        await manager.spawn_agent(
            "OverCapQuarantineChild",
            parent,
            SpawnMandate(parent_did=parent.agent_id),
        )

    assert manager._pending_spawns == 1
    allow_revocation.set()
    await asyncio.wait_for(manager.drain_quarantined_shutdowns(), timeout=1.0)
    assert manager._pending_spawns == 0


@pytest.mark.asyncio
async def test_spawned_by_registration_rehydrates_parent_authority_after_restart(
    tmp_path,
):
    """A fresh manager derives control from the child's durable mandate projection."""

    parent_did = "did:pkh:eip155:1:0xRestartParent"
    child_did = "did:pkh:eip155:1:0xRestartChild"
    parent, mandate = _signed_restored_mandate(
        parent_did,
        child_did,
        purpose="restart regression",
        max_child_depth=1,
        created_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    )
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)

    manager._register_agent("RestartedParent", parent)
    manager._register_agent("RestartedChild", child)
    # Repeating publication of the same object must not duplicate the edge.
    manager._register_agent("RestartedChild", child)

    assert manager.get_children(parent_did) == ["RestartedChild"]
    assert manager.get_mandate("RestartedChild") is mandate
    assert (
        await manager.terminate_child(
            "did:pkh:eip155:1:0xPeer", "RestartedChild"
        )
        is False
    )

    async def remove_registered_child(name, **_kwargs):
        removed = manager._agents.pop(name, None)
        if removed is None:
            return False
        manager._agent_names.pop(child_did, None)
        return True

    manager.remove_agent = AsyncMock(side_effect=remove_registered_child)
    assert await manager.terminate_child(parent_did, "RestartedChild") is True
    assert manager.get_children(parent_did) == []
    assert manager.get_mandate("RestartedChild") is None


@pytest.mark.asyncio
async def test_retained_child_does_not_block_unrelated_registration_after_parent_stop(
    tmp_path,
):
    """A verified projection remains usable after non-cascading withdrawal."""

    parent_did = "did:pkh:eip155:1:0xStoppedParent"
    child_did = "did:pkh:eip155:1:0xRetainedChild"
    parent, mandate = _signed_restored_mandate(parent_did, child_did)
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("StoppedParent", parent)
    manager._register_agent("RetainedChild", child)

    assert await manager.remove_agent("StoppedParent") is True
    assert manager.get_agent("RetainedChild") is child
    assert manager.get_mandate("RetainedChild") is mandate

    unrelated = _make_mock_agent("did:test:unrelated")
    manager._register_agent("Unrelated", unrelated)

    assert manager.get_agent("Unrelated") is unrelated
    assert manager.get_children(parent_did) == ["RetainedChild"]
    assert manager.get_mandate("RetainedChild") is mandate


def test_signed_child_is_not_published_before_parent_authority(tmp_path):
    parent_did = "did:pkh:eip155:1:0xLateParent"
    child_did = "did:pkh:eip155:1:0xEarlyChild"
    parent, mandate = _signed_restored_mandate(parent_did, child_did)
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)

    with pytest.raises(
        RuntimeError, match="before its parent authority is loaded"
    ):
        manager._register_agent("EarlyChild", child)
    assert manager.get_agent("EarlyChild") is None
    assert manager.get_children(parent_did) == []

    manager._register_agent("LateParent", parent)
    manager._register_agent("EarlyChild", child)
    assert manager.get_children(parent_did) == ["EarlyChild"]


def test_expired_signed_child_is_never_published(tmp_path):
    parent_did = "did:pkh:eip155:1:0xExpiredParent"
    child_did = "did:pkh:eip155:1:0xExpiredChild"
    parent, mandate = _signed_restored_mandate(
        parent_did,
        child_did,
        ttl_seconds=1,
        created_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
    )
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("ExpiredParent", parent)

    with pytest.raises(RuntimeError, match="mandate has expired"):
        manager._register_agent("ExpiredChild", child)

    assert manager.get_agent("ExpiredChild") is None
    assert manager.get_children(parent_did) == []
    assert manager.get_mandate("ExpiredChild") is None


@pytest.mark.asyncio
async def test_spawned_by_restart_expiry_writes_auto_discovery_retirement(
    tmp_path,
):
    """A signed child that expires while down is retired after one cold boot."""

    parent_did = "did:pkh:eip155:1:0xOfflineExpiryParent"
    child_did = "did:pkh:eip155:1:0xOfflineExpiryChild"
    parent, mandate = _signed_restored_mandate(
        parent_did,
        child_did,
        ttl_seconds=1,
        created_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
    )
    agent_dir = tmp_path / "agent_data" / "OfflineExpiryChild"
    agent_dir.mkdir(parents=True)
    (agent_dir / "kestrel_prime.db").touch()
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("OfflineExpiryParent", parent)

    class ExpiredHostedChild:
        def __init__(self, *, did, **_kwargs):
            self.agent_id = did
            self.did = did
            self.identity = None
            self._persisted_spawn_mandate = mandate

        async def initialize(self):
            await self._host_authority_preflight(self)

        async def shutdown(self):
            return None

    with (
        patch.object(LocalAgentConfig, "validate_runtime", return_value=[]),
        patch(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            new=AsyncMock(return_value=child_did),
        ),
        patch(
            "kestrel_sovereign.multi_agent.agent_manager.KestrelAgent",
            ExpiredHostedChild,
        ),
        pytest.raises(RuntimeError, match="mandate has expired"),
    ):
        await manager.load_agent(
            "OfflineExpiryChild",
            LocalAgentConfig(
                data_dir=Path("agent_data") / "OfflineExpiryChild",
                port=8801,
            ),
        )

    assert manager.get_agent("OfflineExpiryChild") is None
    assert (agent_dir / ".kestrel-spawn-retired").read_text().strip() == child_did


@pytest.mark.asyncio
async def test_configured_retired_spawn_is_not_selected_for_restart(tmp_path):
    """A persisted startup row cannot bypass its durable retirement denial."""

    manager = AgentManager(base_data_dir=tmp_path)
    manager._initialize_agent = AsyncMock()
    config = MultiAgentConfig(
        agents={
            "RetiredChild": LocalAgentConfig(
                data_dir="agent_data/RetiredChild",
                port=8801,
            )
        }
    )

    with patch(
        "kestrel_sovereign.multi_agent.agent_manager."
        "spawn_retirement_denies_startup",
        return_value=True,
    ) as retired:
        assert await manager.load_from_config(config) == 0

    retired.assert_called_once_with(
        (tmp_path / "agent_data" / "RetiredChild").resolve()
    )
    manager._initialize_agent.assert_not_awaited()


def test_cold_restore_enforces_current_spawn_cap(tmp_path):
    parent_did = "did:pkh:eip155:1:0xCappedParent"
    first_did = "did:pkh:eip155:1:0xCappedFirst"
    second_did = "did:pkh:eip155:1:0xCappedSecond"
    private_key, _ = generate_secp256k1_keypair()
    parent = _make_mock_agent(parent_did)
    parent._private_key = private_key
    parent.identity = None
    first = _make_mock_agent(first_did)
    first._persisted_spawn_mandate = sign_mandate(
        SpawnMandate(parent_did=parent_did, child_did=first_did),
        private_key,
    )
    second = _make_mock_agent(second_did)
    second._persisted_spawn_mandate = sign_mandate(
        SpawnMandate(parent_did=parent_did, child_did=second_did),
        private_key,
    )
    manager = AgentManager(base_data_dir=tmp_path)
    manager._max_spawned_agents = 1
    manager._register_agent("CappedParent", parent)
    manager._register_agent("CappedFirst", first)

    with pytest.raises(RuntimeError, match="spawned-agent cap"):
        manager._register_agent("CappedSecond", second)

    assert manager.get_children(parent_did) == ["CappedFirst"]
    assert manager.get_agent("CappedSecond") is None
    assert len(manager._child_mandates) == 1


def test_cold_restore_counts_quarantined_spawn_cap_reservation(tmp_path):
    """A retained failed-cleanup slot remains capacity even without a mandate."""

    parent_did = "did:pkh:eip155:1:0xQuarantineParent"
    child_did = "did:pkh:eip155:1:0xRestoredAroundQuarantine"
    parent, mandate = _signed_restored_mandate(parent_did, child_did)
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._max_spawned_agents = 1
    manager._register_agent("QuarantineParent", parent)
    # A quarantined cleanup transfers its live cap slot by clearing the
    # admission's ``spawn_slot_active`` while deliberately retaining this
    # durable count until revocation and runtime cleanup both settle.
    manager._pending_spawns = 1

    with pytest.raises(RuntimeError, match="spawned-agent cap"):
        manager._register_agent("RestoredAroundQuarantine", child)

    assert manager.get_agent("RestoredAroundQuarantine") is None
    assert manager.get_mandate("RestoredAroundQuarantine") is None
    assert manager.get_children(parent_did) == []


def test_pre_registry_restore_counts_unloaded_durable_witness_at_cap(tmp_path):
    """Backfilling one old receipt cannot displace a stopped durable child."""

    cold_name = "StoppedDurableChild"
    cold_did = "did:test:stopped-durable-child"
    registry = SpawnAuthorityRegistry(tmp_path)
    registry.record_active(
        child_name=cold_name,
        child_did=cold_did,
        mandate=SpawnMandate(
            parent_did="did:test:stopped-durable-parent",
            child_did=cold_did,
            ttl_seconds=0,
            parent_signature="signed-stopped-durable",
        ),
        config=LocalAgentConfig(
            data_dir=Path("agent_data") / cold_name,
            port=8802,
            autostart=False,
        ),
    )
    parent_did = "did:pkh:eip155:1:0xBackfillCapParent"
    candidate_did = "did:pkh:eip155:1:0xBackfillCapChild"
    parent, mandate = _signed_restored_mandate(parent_did, candidate_did)
    candidate = _make_mock_agent(candidate_did)
    candidate._persisted_spawn_mandate = mandate
    candidate_config = LocalAgentConfig(
        data_dir=Path("agent_data") / "BackfillCapChild",
        port=8803,
        autostart=False,
    )
    manager = AgentManager(base_data_dir=tmp_path)
    manager._max_spawned_agents = 1
    manager._register_agent("BackfillCapParent", parent)
    manager._created_configs["BackfillCapChild"] = candidate_config

    with pytest.raises(RuntimeError, match="spawned-agent cap"):
        manager._verify_agent_authority("BackfillCapChild", candidate)

    assert registry.get(candidate_did) is None
    assert {witness.child_did for witness in registry.records()} == {cold_did}


@pytest.mark.asyncio
async def test_pre_registry_finite_restore_keeps_expiry_owner_after_boot_failure(
    tmp_path,
    monkeypatch,
):
    """Backfilled authority keeps its signed deadline without a published child."""

    parent_did = "did:test:backfilled-expiry-parent"
    child_did = "did:test:backfilled-expiry-child"
    parent, mandate = _signed_restored_mandate(
        parent_did,
        child_did,
        ttl_seconds=3600,
    )
    child_name = "BackfilledExpiryChild"
    candidate = _make_mock_agent(child_did)
    candidate._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("BackfilledExpiryParent", parent)
    manager._created_configs[child_name] = LocalAgentConfig(
        data_dir=Path("agent_data") / child_name,
        port=8802,
        autostart=False,
    )
    monkeypatch.setattr(
        SpawnedAgentLifecycle,
        "_remaining_ttl_seconds",
        staticmethod(lambda *_args: 0.0),
    )

    manager._verify_agent_authority(child_name, candidate)
    await manager._discard_unpublished_initialized_agent(child_name, candidate)
    for _ in range(20):
        witness = manager._spawn_authority_registry.get(child_did)
        lifecycle = getattr(manager, "_lifecycle", None)
        if (
            witness is not None
            and witness.retired
            and isinstance(lifecycle, SpawnedAgentLifecycle)
            and not lifecycle._cold_ttl_tasks
        ):
            break
        await asyncio.sleep(0)

    witness = manager._spawn_authority_registry.get(child_did)
    assert witness is not None and witness.retired


def test_hybrid_parent_signing_alias_restores_to_stable_parent(tmp_path):
    from kestrel_sovereign.identity.did_web import build_verification_methods
    from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair

    legacy_did = "did:pkh:eip155:1:0xHybridParent"
    signing_did = "did:web:example.test:hybrid-parent"
    child_did = "did:pkh:eip155:1:0xHybridChild"
    keypair = generate_hybrid_keypair()
    identity = SimpleNamespace(
        is_hybrid=True,
        legacy_did=legacy_did,
        new_did=signing_did,
        signing_did=signing_did,
        hybrid_keypair=keypair,
        new_verification_methods=build_verification_methods(
            signing_did,
            keypair.public_keys(),
        ),
    )
    parent = _make_mock_agent(legacy_did)
    parent._private_key = None
    parent.identity = identity
    child = _make_mock_agent(child_did)
    mandate = sign_mandate(
        SpawnMandate(parent_did=signing_did, child_did=child_did),
        None,
        parent_identity=identity,
    )
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)

    ordered = manager._registration_order_for_initialized_agents(
        [("HybridChild", object(), child), ("HybridParent", object(), parent)]
    )
    assert [item[0] for item in ordered] == ["HybridParent", "HybridChild"]
    manager._register_agent("HybridParent", parent)
    manager._register_agent("HybridChild", child)

    assert manager.get_children(legacy_did) == ["HybridChild"]
    assert manager.get_mandate("HybridChild") is mandate


def test_cold_restore_rejects_unwitnessed_receipt_claiming_pre_rotation(
    tmp_path,
    post_ceremony_material,
):
    """The signed timestamp alone cannot prove pre-rotation issuance."""

    from kestrel_sovereign.identity.runtime_identity import load_agent_identity

    identity = load_agent_identity(
        post_ceremony_material.legacy_key_id,
        storage_dir=post_ceremony_material.storage_dir,
    )
    legacy_keypair = post_ceremony_material.load_legacy_keypair()
    cutoff = datetime.fromisoformat(identity.succession_statement.effective_from)
    child_did = "did:pkh:eip155:1:0xPreRotationChild"
    mandate = sign_mandate(
        SpawnMandate(
            parent_did=identity.legacy_did,
            child_did=child_did,
            ttl_seconds=0,
            created_at=(cutoff - timedelta(seconds=1)).isoformat(),
        ),
        legacy_keypair.private_key,
    )

    parent = _make_mock_agent(identity.legacy_did)
    parent._private_key = None
    parent.identity = identity
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)

    manager._register_agent("RotatedParent", parent)
    with pytest.raises(
        RuntimeError,
        match="Persisted spawn mandate signature is invalid",
    ):
        manager._register_agent("PersistentChild", child)

    assert manager.get_children(identity.legacy_did) == []
    assert manager.get_mandate("PersistentChild") is None


@pytest.mark.asyncio
async def test_new_spawn_accepts_hybrid_parent_signing_alias(tmp_path):
    legacy_did = "did:pkh:eip155:1:0xRotatedParent"
    signing_did = "did:web:example.test:rotated-parent"
    parent = _make_mock_agent(legacy_did)
    parent.identity = SimpleNamespace(
        is_hybrid=True,
        legacy_did=legacy_did,
        new_did=signing_did,
    )
    parent.features = {}
    child = _make_mock_agent("did:test:normalized-child")
    manager = AgentManager(base_data_dir=tmp_path)
    manager._do_spawn = AsyncMock(return_value=child)
    mandate = SpawnMandate(parent_did=signing_did)

    result = await manager.spawn_agent("NormalizedChild", parent, mandate)

    assert result is child
    captured = manager._do_spawn.await_args.args[2]
    assert mandate.parent_did == signing_did
    assert captured is not mandate
    assert captured.parent_did == legacy_did


@pytest.mark.asyncio
async def test_unsigned_spawned_by_projection_never_restores_governance(tmp_path):
    parent_did = "did:pkh:eip155:1:0xLegacyParent"
    child_did = "did:pkh:eip155:1:0xLegacyChild"
    parent = _make_mock_agent(parent_did)
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = SpawnMandate(
        parent_did=parent_did,
        child_did=child_did,
    )
    manager = AgentManager(base_data_dir=tmp_path)

    manager._register_agent("LegacyParent", parent)
    manager._register_agent("LegacyChild", child)

    assert manager.get_children(parent_did) == []
    assert manager.get_mandate("LegacyChild") is None
    manager._do_spawn = AsyncMock()

    with pytest.raises(ValueError, match="does not grant delegation authority"):
        await manager.spawn_agent(
            "Grandchild",
            child,
            SpawnMandate(parent_did=child_did),
        )

    manager._do_spawn.assert_not_awaited()


def test_signed_descendant_cannot_restore_through_non_governing_parent(tmp_path):
    """An unsigned restored child cannot regain delegation by signing a child."""

    legacy_parent_did = "did:pkh:eip155:1:0xUnsignedRestoredParent"
    descendant_did = "did:pkh:eip155:1:0xSignedDescendant"
    legacy_parent = _make_mock_agent(legacy_parent_did)
    legacy_parent._persisted_spawn_mandate = SpawnMandate(
        parent_did="did:pkh:eip155:1:0xLegacyAncestor",
        child_did=legacy_parent_did,
    )
    private_key, _ = generate_secp256k1_keypair()
    legacy_parent._private_key = private_key
    legacy_parent.identity = None
    descendant = _make_mock_agent(descendant_did)
    descendant._persisted_spawn_mandate = sign_mandate(
        SpawnMandate(
            parent_did=legacy_parent_did,
            child_did=descendant_did,
        ),
        private_key,
    )
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("UnsignedRestoredParent", legacy_parent)

    with pytest.raises(RuntimeError, match="non-governing parent"):
        manager._register_agent("SignedDescendant", descendant)

    assert manager.get_agent("SignedDescendant") is None
    assert manager.get_mandate("SignedDescendant") is None


@pytest.mark.asyncio
async def test_failed_private_hosted_spawn_is_retained_for_runtime_offboarding(
    tmp_path,
):
    manager = AgentManager(base_data_dir=tmp_path)
    did = "did:pkh:failed-private-hosted-spawn"
    scope = resolve_isolated_runtime_namespace(
        manager._isolated_runtime_root,
        derive_isolated_runtime_namespace(did),
    )
    prepare_isolated_runtime_namespace(scope, did)
    credential = scope.path / "credential"
    credential.write_text("must-not-survive-failed-spawn")
    child = _make_mock_agent(did)
    child.did = did
    child.isolated_runtime_scope = scope
    manager._initialize_agent = AsyncMock(return_value=child)
    manager._on_agent_registered = AsyncMock(
        side_effect=RuntimeError("host onboarding failed")
    )
    admission, owns = await manager._admit_agent_operation(
        "FailedHostedChild",
        kind="spawn",
    )
    assert owns

    async def bind_private_candidate(candidate):
        admission.child = candidate

    admission.before_publish = bind_private_candidate
    admission.before_publish_rollback = AsyncMock()
    try:
        with pytest.raises(RuntimeError, match="host onboarding failed"):
            await manager.load_agent(
                "FailedHostedChild",
                LocalAgentConfig(data_dir="unused", port=8801),
            )

        assert admission.unpublished_cleanup_deferred_to_spawn is True
        child.shutdown.assert_not_awaited()
        assert credential.exists()

        await manager._rollback_uncommitted_spawn_runtime(admission, child)

        child.shutdown.assert_awaited_once_with()
        assert not scope.path.exists()
        assert admission.unpublished_cleanup_deferred_to_spawn is False
    finally:
        await manager._release_agent_operation(admission)


def test_tampered_signed_lineage_fails_closed_and_rolls_back_parent_load(tmp_path):
    parent_did = "did:pkh:eip155:1:0xTamperParent"
    child_did = "did:pkh:eip155:1:0xTamperChild"
    parent, mandate = _signed_restored_mandate(parent_did, child_did)
    mandate.max_child_depth += 1
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("TamperParent", parent)

    with pytest.raises(RuntimeError, match="signature is invalid"):
        manager._register_agent("TamperChild", child)

    assert manager.get_agent("TamperParent") is parent
    assert manager.get_agent("TamperChild") is None
    assert manager.get_children(parent_did) == []


def test_failed_onboarding_rolls_back_rehydrated_parent_authority(tmp_path):
    parent_did = "did:pkh:eip155:1:0xRollbackParent"
    child_did = "did:pkh:eip155:1:0xRollbackChild"
    parent, mandate = _signed_restored_mandate(
        parent_did,
        child_did,
        purpose="rollback regression",
        created_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    )
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)

    manager._register_agent("RollbackParent", parent)
    manager._register_agent("RollbackChild", child)
    manager._withdraw_initialized_agent("RollbackChild", child)

    assert manager.get_children(parent_did) == []
    assert manager.get_mandate("RollbackChild") is None


@pytest.mark.asyncio
async def test_duplicate_did_rollback_preserves_live_unsigned_lineage_guard(
    tmp_path,
):
    """A rejected candidate cannot erase another agent's spawn restriction."""

    shared_did = "did:test:shared-unsigned-lineage"
    live = _make_mock_agent(shared_did)
    live._persisted_spawn_mandate = SpawnMandate(
        parent_did="did:test:legacy-parent",
        child_did=shared_did,
        parent_signature=None,
    )
    duplicate = _make_mock_agent(shared_did)
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("LiveUnsigned", live)

    with pytest.raises(RuntimeError, match="already routed"):
        manager._register_agent("Duplicate", duplicate)
    manager._withdraw_initialized_agent("Duplicate", duplicate)

    assert manager.get_agent("LiveUnsigned") is live
    assert shared_did in manager._non_governing_spawn_lineage
    manager._do_spawn = AsyncMock(
        side_effect=AssertionError("unsigned lineage reached spawn I/O")
    )
    with pytest.raises(ValueError, match="unsigned restored lineage"):
        await manager.spawn_agent(
            "ForbiddenDescendant",
            live,
            SpawnMandate(parent_did=shared_did),
        )
    manager._do_spawn.assert_not_awaited()


def test_duplicate_signed_preparation_never_clears_live_unsigned_lineage_guard(
    tmp_path,
):
    """The private preparation window must reject a published DID alias."""

    shared_did = "did:test:shared-preparation-lineage"
    live = _make_mock_agent(shared_did)
    live._persisted_spawn_mandate = SpawnMandate(
        parent_did="did:test:legacy-preparation-parent",
        child_did=shared_did,
        parent_signature=None,
    )
    parent, signed = _signed_restored_mandate(
        "did:test:signed-preparation-parent",
        shared_did,
    )
    duplicate = _make_mock_agent(shared_did)
    duplicate._persisted_spawn_mandate = signed
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("LegacyChild", live)
    manager._register_agent("SignedParent", parent)

    with pytest.raises(RuntimeError, match="already routed"):
        manager._prepare_agent_authority("SignedDuplicate", duplicate)

    assert shared_did in manager._non_governing_spawn_lineage
    assert manager.get_mandate("SignedDuplicate") is None


@pytest.mark.asyncio
async def test_spawn_commit_rechecks_cap_after_concurrent_authority_restore(tmp_path):
    """A cold-loaded child can consume the last cap slot during spawn I/O."""

    parent = _make_mock_agent("did:pkh:eip155:1:0xSpawnParent")
    parent._private_key, _ = generate_secp256k1_keypair()
    parent.identity = None
    parent.features = {}
    fresh = _make_mock_agent("did:pkh:eip155:1:0xFreshChild")
    restored = _make_mock_agent("did:pkh:eip155:1:0xRestoredChild")
    other_parent, restored_mandate = _signed_restored_mandate(
        "did:pkh:eip155:1:0xOtherParent",
        restored.agent_id,
        purpose="cold load won the slot",
    )
    restored._persisted_spawn_mandate = restored_mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._max_spawned_agents = 1
    manager._register_agent("OtherParent", other_parent)

    async def create_after_restore(name, **kwargs):
        manager._register_agent("RestoredChild", restored)
        await _persist_and_publish_spawn_test_child(manager, name, fresh, kwargs)
        return fresh

    async def rollback_fresh(_admission, _child):
        manager._agents.pop("FreshChild", None)
        manager._agent_names.pop(fresh.agent_id, None)
        return False

    manager.create_agent = create_after_restore
    manager._rollback_uncommitted_spawn = rollback_fresh

    with pytest.raises(ValueError, match="spawned-agent cap"):
        await manager.spawn_agent(
            "FreshChild",
            parent,
            SpawnMandate(parent_did=parent.agent_id, purpose="racing spawn"),
        )

    assert manager.get_children(restored._persisted_spawn_mandate.parent_did) == [
        "RestoredChild"
    ]
    assert manager.get_agent("FreshChild") is None
    assert manager._pending_spawns == 0


@pytest.mark.asyncio
async def test_spawn_cap_wait_expires_at_signed_deadline(tmp_path):
    """A cap-loser's blocked cleanup cannot extend another child's TTL."""

    parent = _make_mock_agent("did:pkh:eip155:1:0xDeadlineWaitParent")
    parent._private_key, _ = generate_secp256k1_keypair()
    parent.identity = None
    parent.features = {}
    first = _make_mock_agent("did:pkh:eip155:1:0xDeadlineWaitFirst")
    second = _make_mock_agent("did:pkh:eip155:1:0xDeadlineWaitSecond")
    restored = _make_mock_agent("did:pkh:eip155:1:0xDeadlineWaitRestored")
    other_parent, restored_mandate = _signed_restored_mandate(
        "did:pkh:eip155:1:0xDeadlineWaitOtherParent",
        restored.agent_id,
        purpose="cold load consumed a slot",
    )
    restored._persisted_spawn_mandate = restored_mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._max_spawned_agents = 2
    manager._register_agent("DeadlineWaitOtherParent", other_parent)
    children = {
        "DeadlineWaitFirst": first,
        "DeadlineWaitSecond": second,
    }
    both_published = asyncio.Event()
    published: set[str] = set()
    release_first_rollback = asyncio.Event()
    first_rollback_started = asyncio.Event()

    async def create_after_restore(name, **kwargs):
        child = children[name]
        await _persist_and_publish_spawn_test_child(manager, name, child, kwargs)
        published.add(name)
        if len(published) == 2:
            manager._register_agent("DeadlineWaitRestored", restored)
            both_published.set()
        await both_published.wait()
        if name == "DeadlineWaitSecond":
            await first_rollback_started.wait()
        return child

    async def rollback_fresh(admission, child):
        if admission.name == "DeadlineWaitFirst":
            first_rollback_started.set()
            cleanup = asyncio.create_task(release_first_rollback.wait())
            await await_lifecycle_task_completion(cleanup)
        manager._agents.pop(admission.name, None)
        manager._agent_names.pop(child.agent_id, None)
        return False

    manager.create_agent = create_after_restore
    manager._rollback_uncommitted_spawn = rollback_fresh
    first_spawn = asyncio.create_task(
        manager.spawn_agent(
            "DeadlineWaitFirst",
            parent,
            SpawnMandate(parent_did=parent.agent_id, ttl_seconds=1),
        )
    )
    second_spawn = asyncio.create_task(
        manager.spawn_agent(
            "DeadlineWaitSecond",
            parent,
            SpawnMandate(parent_did=parent.agent_id, ttl_seconds=1),
        )
    )
    try:
        await asyncio.wait_for(first_rollback_started.wait(), timeout=1.0)
        with pytest.raises(
            PersistedSpawnMandateExpiredError,
            match="expired during active host admission",
        ):
            await asyncio.wait_for(second_spawn, timeout=1.5)
        assert manager.get_agent("DeadlineWaitSecond") is None
    finally:
        release_first_rollback.set()
        results = await asyncio.gather(
            first_spawn,
            second_spawn,
            return_exceptions=True,
        )
    assert isinstance(results[0], ValueError)


@pytest.mark.asyncio
async def test_failed_spawn_rollback_revokes_authority_but_keeps_restrictions(tmp_path):
    private_key, _ = generate_secp256k1_keypair()
    parent = _make_mock_agent("did:test:rollback-parent")
    parent._private_key = private_key
    parent.identity = None
    parent.features = {}
    graph = SimpleNamespace(
        add_trusted_cross_agent_edge=AsyncMock(),
        delete_edge=AsyncMock(),
    )
    child = _make_mock_agent("did:test:rollback-child")
    child._raw_storage = SimpleNamespace(graph=graph)
    manager = AgentManager(base_data_dir=tmp_path)

    async def create_child(name, **kwargs):
        await _persist_and_publish_spawn_test_child(manager, name, child, kwargs)
        return child

    async def remove_child(name, **_kwargs):
        manager._agents.pop(name, None)
        manager._agent_names.pop(child.agent_id, None)
        return True

    manager.create_agent = AsyncMock(side_effect=create_child)
    manager.remove_agent = AsyncMock(side_effect=remove_child)
    manager._apply_delegated_budget = AsyncMock(
        side_effect=RuntimeError("budget provider failed")
    )

    with pytest.raises(RuntimeError, match="budget provider failed"):
        await manager.spawn_agent(
            "RollbackChild",
            parent,
            SpawnMandate(parent_did=parent.agent_id, purpose="must roll back"),
        )

    assert graph.add_trusted_cross_agent_edge.await_count == 2
    revoked = graph.add_trusted_cross_agent_edge.await_args_list[-1]
    assert revoked.args == (child.agent_id, parent.agent_id, "spawned_by")
    assert revoked.kwargs["properties"]["parent_signature"] is None
    assert child._persisted_spawn_mandate.parent_signature is None
    graph.delete_edge.assert_not_awaited()


@pytest.mark.asyncio
async def test_spawn_restamps_proposal_at_final_child_identity(tmp_path):
    private_key, _ = generate_secp256k1_keypair()
    parent = _make_mock_agent("did:test:ttl-parent")
    parent._private_key = private_key
    parent.identity = None
    parent.features = {}
    graph = SimpleNamespace(
        add_trusted_cross_agent_edge=AsyncMock(),
        delete_edge=AsyncMock(),
    )
    child = _make_mock_agent("did:test:ttl-child")
    child._raw_storage = SimpleNamespace(graph=graph)
    manager = AgentManager(base_data_dir=tmp_path)

    async def create_child(name, **kwargs):
        await _persist_and_publish_spawn_test_child(manager, name, child, kwargs)
        return child

    manager.create_agent = AsyncMock(side_effect=create_child)
    old_created_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    mandate = SpawnMandate(
        parent_did=parent.agent_id,
        ttl_seconds=60,
        created_at=old_created_at,
    )

    await manager.spawn_agent("TTLChild", parent, mandate)

    persisted = child._persisted_spawn_mandate
    assert mandate.created_at == old_created_at
    assert persisted is not mandate
    assert persisted.created_at != old_created_at
    assert remaining_spawn_ttl_seconds(persisted.created_at, 60) > 59


@pytest.mark.asyncio
async def test_live_spawn_owns_signed_deadline_before_delegated_budget(tmp_path):
    """The first post-signature await must already have an expiry owner."""

    private_key, _ = generate_secp256k1_keypair()
    parent = _make_mock_agent("did:test:live-deadline-parent")
    parent._private_key = private_key
    parent.identity = None
    parent.features = {}
    graph = SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
    child = _make_mock_agent("did:test:live-deadline-child")
    child._raw_storage = SimpleNamespace(graph=graph)
    manager = AgentManager(base_data_dir=tmp_path)
    deadline_owned_during_budget = []

    async def create_child(name, **kwargs):
        await _persist_and_publish_spawn_test_child(manager, name, child, kwargs)
        return child

    async def observe_budget(_name, _parent, candidate, _mandate, **_kwargs):
        deadline_owned_during_budget.append(
            vars(candidate).get("_host_authority_boot_deadline_handle") is not None
        )

    manager.create_agent = AsyncMock(side_effect=create_child)
    manager._apply_delegated_budget = AsyncMock(side_effect=observe_budget)

    await manager.spawn_agent(
        "LiveDeadlineChild",
        parent,
        SpawnMandate(parent_did=parent.agent_id, ttl_seconds=60),
    )

    assert deadline_owned_during_budget == [True]
    assert vars(child).get("_host_authority_boot_deadline_handle") is None


@pytest.mark.asyncio
async def test_live_signed_deadline_cancels_stalled_budget_and_retires(tmp_path):
    """A stalled custody provider cannot keep signed authority alive."""

    private_key, _ = generate_secp256k1_keypair()
    parent = _make_mock_agent("did:test:stalled-budget-parent")
    parent._private_key = private_key
    parent.identity = None
    parent.features = {}
    graph = SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
    child = _make_mock_agent("did:test:stalled-budget-child")
    child._raw_storage = SimpleNamespace(graph=graph)
    manager = AgentManager(base_data_dir=tmp_path)
    budget_cancelled = asyncio.Event()

    async def create_child(name, **kwargs):
        await _persist_and_publish_spawn_test_child(manager, name, child, kwargs)
        return child

    async def stalled_budget(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            budget_cancelled.set()

    manager.create_agent = AsyncMock(side_effect=create_child)
    manager._apply_delegated_budget = AsyncMock(side_effect=stalled_budget)
    with patch(
        "kestrel_sovereign.kestrel_agent.remaining_spawn_ttl_seconds",
        return_value=0.03,
    ):
        spawn = asyncio.create_task(
            manager.spawn_agent(
                "StalledBudgetChild",
                parent,
                SpawnMandate(parent_did=parent.agent_id, ttl_seconds=1),
            )
        )
        with pytest.raises(
            PersistedSpawnMandateExpiredError,
            match="expired during active host admission",
        ):
            await asyncio.wait_for(spawn, timeout=0.5)

    assert budget_cancelled.is_set()
    assert vars(child).get("_host_authority_boot_deadline_handle") is None
    witness = manager._spawn_authority_registry.get(child.agent_id)
    assert witness is not None and witness.retired
    assert graph.add_trusted_cross_agent_edge.await_count == 2
    assert (
        graph.add_trusted_cross_agent_edge.await_args_list[-1]
        .kwargs["properties"]["parent_signature"]
        is None
    )


@pytest.mark.asyncio
async def test_live_signed_deadline_maps_grouped_readiness_cancellation_to_expiry(
    tmp_path,
):
    """Ready rollback cancellation groups remain ordinary mandate expiry."""

    private_key, _ = generate_secp256k1_keypair()
    parent = _make_mock_agent("did:test:grouped-readiness-parent")
    parent._private_key = private_key
    parent.identity = None
    parent.features = {}
    graph = SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
    child = _make_mock_agent("did:test:grouped-readiness-child")
    child._raw_storage = SimpleNamespace(graph=graph)
    manager = AgentManager(base_data_dir=tmp_path)

    async def expire_during_readiness(name, **kwargs):
        await _persist_and_publish_spawn_test_child(manager, name, child, kwargs)
        child._host_authority_boot_expired = True
        raise BaseExceptionGroup(
            "readiness and rollback observed deadline cancellation",
            [asyncio.CancelledError(), asyncio.CancelledError()],
        )

    manager.create_agent = AsyncMock(side_effect=expire_during_readiness)

    with pytest.raises(
        PersistedSpawnMandateExpiredError,
        match="expired during active host admission",
    ):
        await manager.spawn_agent(
            "GroupedReadinessChild",
            parent,
            SpawnMandate(parent_did=parent.agent_id, ttl_seconds=60),
        )

    assert manager.get_agent("GroupedReadinessChild") is None
    witness = manager._spawn_authority_registry.get(child.agent_id)
    assert witness is not None and witness.retired
    assert graph.add_trusted_cross_agent_edge.await_count == 2


@pytest.mark.asyncio
async def test_spawn_expired_before_commit_rolls_back_signed_receipt(tmp_path):
    private_key, _ = generate_secp256k1_keypair()
    parent = _make_mock_agent("did:test:deadline-parent")
    parent._private_key = private_key
    parent.identity = None
    parent.features = {}
    graph = SimpleNamespace(
        add_trusted_cross_agent_edge=AsyncMock(),
        delete_edge=AsyncMock(),
    )
    child = _make_mock_agent("did:test:deadline-child")
    child._raw_storage = SimpleNamespace(graph=graph)
    manager = AgentManager(base_data_dir=tmp_path)

    async def create_child(name, **kwargs):
        await _persist_and_publish_spawn_test_child(manager, name, child, kwargs)
        return child

    async def remove_child(name, **_kwargs):
        manager._agents.pop(name, None)
        manager._agent_names.pop(child.agent_id, None)
        return True

    manager.create_agent = AsyncMock(side_effect=create_child)
    manager.remove_agent = AsyncMock(side_effect=remove_child)

    proposal = SpawnMandate(parent_did=parent.agent_id, ttl_seconds=1)
    proposal_created_at = proposal.created_at
    with (
        patch(
            "kestrel_sovereign.multi_agent.agent_manager.remaining_spawn_ttl_seconds",
            return_value=0,
        ),
        pytest.raises(RuntimeError, match="expired before governance commit"),
    ):
        await manager.spawn_agent("DeadlineChild", parent, proposal)

    assert graph.add_trusted_cross_agent_edge.await_count == 2
    revoked = graph.add_trusted_cross_agent_edge.await_args_list[-1]
    assert revoked.kwargs["properties"]["parent_signature"] is None
    assert revoked.kwargs["properties"]["created_at"] == proposal_created_at
    assert child._persisted_spawn_mandate.created_at == proposal_created_at


@pytest.mark.asyncio
async def test_restored_authority_rehydrates_lifecycle_for_parent_termination(
    tmp_path,
):
    """Cold load itself must create lifecycle state and arm the original TTL."""

    parent_did = "did:pkh:eip155:1:0xLifecycleParent"
    child_did = "did:pkh:eip155:1:0xLifecycleChild"
    created_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    parent, mandate = _signed_restored_mandate(
        parent_did,
        child_did,
        purpose="restore lifecycle",
        ttl_seconds=3600,
        created_at=created_at,
    )
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("LifecycleParent", parent)
    manager._register_agent("LifecycleChild", child)

    lifecycle = manager._lifecycle

    assert isinstance(lifecycle, SpawnedAgentLifecycle)
    assert lifecycle.is_tracked("LifecycleChild")
    assert lifecycle._tracked["LifecycleChild"].started_at == created_at
    assert lifecycle._tracked["LifecycleChild"].ttl_task is not None

    async def remove_registered_child(name, **_kwargs):
        removed = manager._agents.pop(name, None)
        if removed is None:
            return False
        manager._agent_names.pop(child_did, None)
        return True

    manager.remove_agent = AsyncMock(side_effect=remove_registered_child)
    result = await lifecycle.terminate("LifecycleChild")

    assert result is not None
    assert result.started_at == created_at
    assert manager.get_children(parent_did) == []


def test_cold_budgeted_child_is_not_published_without_restored_custody(tmp_path):
    parent_did = "did:pkh:eip155:1:0xBudgetParent"
    child_did = "did:pkh:eip155:1:0xBudgetChild"
    parent, mandate = _signed_restored_mandate(
        parent_did,
        child_did,
        budget_allocation=5,
    )
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)

    manager._register_agent("BudgetParent", parent)
    with pytest.raises(RuntimeError, match="delegated wallet custody"):
        manager._register_agent("BudgetChild", child)

    assert manager.get_agent("BudgetChild") is None
    assert manager.get_children(parent_did) == []
    assert manager.get_mandate("BudgetChild") is None


def test_expired_cold_budgeted_receipt_precedes_custody_refusal(tmp_path):
    parent_did = "did:pkh:eip155:1:0xExpiredBudgetParent"
    child_did = "did:pkh:eip155:1:0xExpiredBudgetChild"
    parent, mandate = _signed_restored_mandate(
        parent_did,
        child_did,
        budget_allocation=5,
        ttl_seconds=1,
        created_at=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
    )
    child = _make_mock_agent(child_did)
    child._persisted_spawn_mandate = mandate
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("ExpiredBudgetParent", parent)

    with pytest.raises(PersistedSpawnMandateExpiredError):
        manager._verify_agent_authority("ExpiredBudgetChild", child)


def test_registration_refuses_cyclic_restored_parent_authority(tmp_path):
    first_did = "did:pkh:eip155:1:0xCycleFirst"
    second_did = "did:pkh:eip155:1:0xCycleSecond"
    first_private, _ = generate_secp256k1_keypair()
    second_private, _ = generate_secp256k1_keypair()
    first = _make_mock_agent(first_did)
    first._private_key = first_private
    first.identity = None
    first._persisted_spawn_mandate = sign_mandate(
        SpawnMandate(parent_did=second_did, child_did=first_did),
        second_private,
    )
    second = _make_mock_agent(second_did)
    second._private_key = second_private
    second.identity = None
    second._persisted_spawn_mandate = sign_mandate(
        SpawnMandate(parent_did=first_did, child_did=second_did),
        first_private,
    )
    manager = AgentManager(base_data_dir=tmp_path)

    ordered = manager._registration_order_for_initialized_agents(
        [
            ("CycleFirst", object(), first),
            ("CycleSecond", object(), second),
        ]
    )
    assert [item[0] for item in ordered] == ["CycleFirst", "CycleSecond"]
    with pytest.raises(RuntimeError, match="parent authority"):
        manager._register_agent("CycleFirst", first)
    with pytest.raises(RuntimeError, match="parent authority"):
        manager._register_agent("CycleSecond", second)

    assert manager.get_children(second_did) == []
    assert manager.get_children(first_did) == []
    assert manager.get_agent("CycleFirst") is None
    assert manager.get_agent("CycleSecond") is None


@pytest.mark.asyncio
async def test_over_cap_spawn_retires_slot_before_rollback_allows_one_winner(
    tmp_path,
):
    """One restored slot rejects one of two pending spawns, not both."""

    manager = AgentManager(base_data_dir=tmp_path)
    manager._max_spawned_agents = 2
    parent = _make_mock_agent("did:pkh:eip155:1:0xConcurrentParent")
    parent._private_key, _ = generate_secp256k1_keypair()
    parent.identity = None
    parent.features = {}
    restored = _make_mock_agent("did:pkh:eip155:1:0xConcurrentRestored")
    other_parent, restored_mandate = _signed_restored_mandate(
        "did:pkh:eip155:1:0xOtherParent",
        restored.agent_id,
    )
    restored._persisted_spawn_mandate = restored_mandate
    manager._register_agent("OtherParent", other_parent)
    fresh = {
        "FirstFresh": _make_mock_agent("did:pkh:eip155:1:0xFirstFresh"),
        "SecondFresh": _make_mock_agent("did:pkh:eip155:1:0xSecondFresh"),
    }
    both_created = asyncio.Event()
    rollback_started = asyncio.Event()
    release_first_rollback = asyncio.Event()
    arrivals = 0
    rollback_names: list[str] = []

    async def create_after_both_reservations(name, **_kwargs):
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            manager._register_agent("ConcurrentRestored", restored)
            both_created.set()
        await both_created.wait()
        child = fresh[name]
        await _persist_and_publish_spawn_test_child(manager, name, child, _kwargs)
        return child

    async def rollback_rejected(admission, child):
        rollback_names.append(admission.name)
        manager._spawn_authority_registry.begin_retirement(
            child_name=admission.name,
            child_did=child.agent_id,
        )
        manager._agents.pop(admission.name, None)
        manager._agent_names.pop(child.agent_id, None)
        if len(rollback_names) == 1:
            rollback_started.set()
            await release_first_rollback.wait()
        return False

    manager.create_agent = create_after_both_reservations
    manager._rollback_uncommitted_spawn = rollback_rejected
    tasks = [
        asyncio.create_task(
            manager.spawn_agent(
                name,
                parent,
                SpawnMandate(parent_did=parent.agent_id, purpose=name),
            )
        )
        for name in fresh
    ]

    await asyncio.wait_for(rollback_started.wait(), timeout=1)
    await asyncio.sleep(0.05)
    release_first_rollback.set()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    assert sum(not isinstance(outcome, BaseException) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
    assert len(manager._child_mandates) == 2
    assert manager._pending_spawns == 0


def _exception_group_contains(
    error: BaseException, expected: type[BaseException]
) -> bool:
    if isinstance(error, expected):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(
            _exception_group_contains(item, expected) for item in error.exceptions
        )
    return False


def _exception_leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        leaves: list[BaseException] = []
        for item in error.exceptions:
            leaves.extend(_exception_leaves(item))
        return leaves
    return [error]


def _admitted_offboarding_failure(error: BaseException) -> AsyncMock:
    """Build a manager mock that marks the real cleanup-admission boundary."""

    async def fail(_name, **kwargs):
        admission = kwargs["offboarding_admission"]
        assert isinstance(admission, RuntimeOffboardingAdmission)
        admission.started = True
        raise error

    return AsyncMock(side_effect=fail)


def test_runtime_offboarding_not_performed_custody_state_is_narrow() -> None:
    ordinary = RuntimeOffboardingNotPerformedError(
        agent_name="Hosted",
        agent_id="did:test:hosted",
        cleanup_state="already_absent",
    )
    unknown = RuntimeOffboardingNotPerformedError(
        agent_name="Hosted",
        agent_id="did:test:hosted",
        cleanup_state="custody_unknown",
    )

    assert ordinary.cleanup_state == ordinary.metadata["runtime_cleanup_state"]
    assert ordinary.metadata["runtime_cleanup_state"] == "already_absent"
    assert ordinary.metadata["runtime_already_absent"] is True
    assert "runtime_retention_unknown" not in ordinary.metadata
    assert unknown.cleanup_state == unknown.metadata["runtime_cleanup_state"]
    assert unknown.metadata["runtime_cleanup_state"] == "custody_unknown"
    assert unknown.metadata["runtime_already_absent"] is False
    assert unknown.metadata["runtime_custody_known"] is False
    assert unknown.metadata["runtime_retention_unknown"] is True
    assert "runtime_retained" not in unknown.metadata
    assert "hosted_runtime_configured" not in unknown.metadata
    with pytest.raises(ValueError, match="invalid runtime offboarding no-op state"):
        RuntimeOffboardingNotPerformedError(
            agent_name="Hosted",
            agent_id="did:test:hosted",
            cleanup_state="removed",
        )


def _admitted_offboarding_success() -> AsyncMock:
    """Build a manager mock which performs the typed admission handshake."""

    async def succeed(_name, **kwargs):
        admission = kwargs["offboarding_admission"]
        assert isinstance(admission, RuntimeOffboardingAdmission)
        admission.started = True
        return True

    return AsyncMock(side_effect=succeed)


def _endpoint_custody_outcome(
    outcome_kind: str,
    *,
    private_path: Path,
) -> BaseException:
    """Build one sanitized public custody shape with private internal causes."""

    retained = RuntimeOffboardingRetainedError(
        agent_name="Hosted",
        agent_id="did:test:restore-conflict",
        runtime_path=private_path.parent,
        cause=OSError(f"private cleanup failure at {private_path}"),
    )
    if outcome_kind == "retained":
        return retained
    if outcome_kind == "not-performed":
        return RuntimeOffboardingNotPerformedError(
            agent_name="Hosted",
            agent_id="did:test:restore-conflict",
            cleanup_state="already_absent",
        )
    if outcome_kind == "grouped-retained":
        return ExceptionGroup(
            "compound offboarding result",
            [
                retained,
                RuntimeError(f"private refund failure at {private_path}"),
            ],
        )
    raise AssertionError(f"unsupported test outcome kind: {outcome_kind}")


@pytest.mark.parametrize("value", ["30s", "nan", "inf", "0", "-1"])
def test_runtime_offboard_timeout_rejects_invalid_values_actionably(value):
    with pytest.raises(
        ValueError,
        match=(
            "^KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S must be finite and positive$"
        ),
    ):
        _parse_runtime_offboard_timeout(value)


@pytest.mark.parametrize(("value", "expected"), [("0.25", 0.25), (30, 30.0)])
def test_runtime_offboard_timeout_accepts_finite_positive_values(value, expected):
    assert _parse_runtime_offboard_timeout(value) == expected


def test_runtime_offboard_timeout_fresh_import_names_malformed_variable():
    env = os.environ.copy()
    env["KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S"] = "30s"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import kestrel_sovereign.multi_agent.agent_manager",
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    diagnostic = completed.stdout + completed.stderr
    assert "KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S" in diagnostic
    assert "could not convert string to float" not in diagnostic


def test_runtime_offboard_timeout_has_independent_thirty_second_default():
    env = os.environ.copy()
    env.pop("KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from kestrel_sovereign.multi_agent.agent_manager import "
                "RUNTIME_OFFBOARD_TIMEOUT_S; print(RUNTIME_OFFBOARD_TIMEOUT_S)"
            ),
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "30.0"
    if "KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S" not in os.environ:
        assert RUNTIME_OFFBOARD_TIMEOUT_S == 30.0


def test_env_example_documents_runtime_offboard_timeout():
    env_example = (Path(__file__).parents[2] / ".env.example").read_text()
    assert "# KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S=30" in env_example
    assert "# Default: 30 seconds" in env_example


class TestAgentManagerBasics:
    """Test AgentManager get/list/remove without real agents."""

    def test_empty_manager(self):
        manager = AgentManager()
        assert manager.list_agents() == {}
        assert manager.get_agent("nonexistent") is None

    def test_get_agent_case_insensitive(self):
        manager = AgentManager()
        mock = _make_mock_agent()
        manager._agents["Claw"] = mock
        manager._agent_names[mock.agent_id] = "Claw"

        assert manager.get_agent("Claw") is mock
        assert manager.get_agent("claw") is mock
        assert manager.get_agent("CLAW") is mock
        assert manager.get_agent("unknown") is None

    def test_list_agents(self):
        manager = AgentManager()
        agent1 = _make_mock_agent("did:1")
        agent2 = _make_mock_agent("did:2")
        manager._agents["Alpha"] = agent1
        manager._agents["Beta"] = agent2

        result = manager.list_agents()
        assert len(result) == 2
        assert result["Alpha"] is agent1
        assert result["Beta"] is agent2

    def test_host_context_registry_binds_existing_agents_and_is_retained(self):
        manager = AgentManager()
        agent1 = _make_mock_agent("did:1")
        agent2 = _make_mock_agent("did:2")
        manager._agents.update({"Alpha": agent1, "Beta": agent2})
        registry = object()

        manager.bind_host_context_clause_registry(registry)

        for agent in (agent1, agent2):
            agent.validate_host_context_clause_registry.assert_called_once_with(
                registry
            )
            agent.bind_host_context_clause_registry.assert_called_once_with(
                registry
            )
            assert (
                agent._host_context_publication_state
                is manager._host_context_publication_state
            )
            assert agent._host_context_publication_generation == 1
        assert manager._host_context_clause_registry is registry
        assert manager._host_context_publication_state.registry is registry
        assert manager._host_context_publication_state.generation == 1

    def test_registration_rebinds_host_context_published_during_initialization(self):
        """A cold agent cannot retain the registry snapshot from construction."""

        manager = AgentManager()
        current_registry = object()
        publication_gate = asyncio.Event()
        # Model the window after construction but before registration: neither
        # fan-out can see the still-unpublished agent.
        manager.bind_host_context_clause_registry(current_registry)
        manager.set_host_context_publication_gate(publication_gate)
        agent = _make_mock_agent("did:cold")

        manager._register_agent("Cold", agent)

        agent.validate_host_context_clause_registry.assert_called_once_with(
            current_registry
        )
        agent.bind_host_context_clause_registry.assert_called_once_with(
            current_registry
        )
        assert agent._host_context_publication_gate is publication_gate
        assert (
            agent._host_context_publication_state
            is manager._host_context_publication_state
        )
        assert (
            agent._host_context_publication_generation
            == manager._host_context_publication_state.generation
        )
        assert manager.get_agent("Cold") is agent

    def test_get_agent_name(self):
        manager = AgentManager()
        mock = _make_mock_agent("did:pkh:test")
        manager._agents["Emma"] = mock
        manager._agent_names["did:pkh:test"] = "Emma"

        assert manager.get_agent_name("did:pkh:test") == "Emma"
        assert manager.get_agent_name("did:unknown") is None

    @pytest.mark.asyncio
    async def test_local_agent_configs_by_did_includes_cold_agents(self, monkeypatch, tmp_path):
        """A host scheduler can resolve both loaded and autostart=false agents."""
        manager = AgentManager(base_data_dir=tmp_path)
        warm = _make_mock_agent("did:pkh:warm")
        manager._agents["Warm"] = warm
        cold_dir = tmp_path / "cold"
        config = MultiAgentConfig(
            agents={
                "Warm": LocalAgentConfig(data_dir="warm", port=8801, autostart=True),
                "Cold": LocalAgentConfig(data_dir="cold", port=8802, autostart=False),
            }
        )

        did_lookup = AsyncMock(return_value="did:pkh:cold")
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            did_lookup,
        )

        mapping = await manager.local_agent_configs_by_did(config)

        assert mapping["did:pkh:warm"][0] == "Warm"
        assert mapping["did:pkh:cold"][0] == "Cold"
        did_lookup.assert_awaited_once_with(
            str(cold_dir),
            mode=AgentDIDLookupMode.COLD_READ_ONLY,
        )

    @pytest.mark.asyncio
    async def test_local_agent_configs_excludes_retired_spawn_from_scheduler_authority(
        self,
        monkeypatch,
        tmp_path,
    ):
        """A retirement denial removes the row from cold-wake authority too."""

        manager = AgentManager(base_data_dir=tmp_path)
        config = MultiAgentConfig(
            agents={
                "RetiredChild": LocalAgentConfig(
                    data_dir="agent_data/RetiredChild",
                    port=8801,
                    autostart=False,
                )
            }
        )
        did_lookup = AsyncMock(return_value="did:test:retired-child")
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            did_lookup,
        )

        with patch(
            "kestrel_sovereign.multi_agent.agent_manager."
            "spawn_retirement_denies_startup",
            return_value=True,
        ) as retired:
            mapping = await manager.local_agent_configs_by_did(config)

        assert mapping == {}
        assert not manager.is_scheduler_agent_authorized("did:test:retired-child")
        retired.assert_called_once_with(
            (tmp_path / "agent_data" / "RetiredChild").resolve()
        )
        did_lookup.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_local_agent_configs_skips_unincepted_cold_agent_but_keeps_healthy_peer(
        self, monkeypatch, tmp_path,
    ):
        """One missing cold identity cannot abort the healthy scheduler fleet."""
        manager = AgentManager(base_data_dir=tmp_path)
        warm = _make_mock_agent("did:pkh:warm")
        manager._agents["Warm"] = warm
        config = MultiAgentConfig(
            agents={
                "Warm": LocalAgentConfig(
                    data_dir="warm", port=8801, autostart=True
                ),
                "Unincepted": LocalAgentConfig(
                    data_dir="unincepted", port=8802, autostart=False
                ),
            }
        )
        missing_identity = RuntimeError("identity database is not initialized")
        did_lookup = AsyncMock(side_effect=missing_identity)
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            did_lookup,
        )

        mapping = await manager.local_agent_configs_by_did(config)

        assert mapping == {"did:pkh:warm": ("Warm", config.agents["Warm"])}
        assert manager.cold_scheduler_identity_failures == [
            ("Unincepted", missing_identity)
        ]
        did_lookup.assert_awaited_once_with(
            str(tmp_path / "unincepted"),
            mode=AgentDIDLookupMode.COLD_READ_ONLY,
        )

    @pytest.mark.asyncio
    async def test_local_agent_configs_skips_unresolved_autostart_agent_but_keeps_healthy_peer(
        self, monkeypatch, tmp_path,
    ):
        """A failed autostart tenant is not scheduler authority for its peer."""
        manager = AgentManager(base_data_dir=tmp_path)
        warm = _make_mock_agent("did:pkh:warm")
        manager._agents["Warm"] = warm
        config = MultiAgentConfig(
            agents={
                "Warm": LocalAgentConfig(data_dir="warm", port=8801),
                "Unresolved": LocalAgentConfig(
                    data_dir="unresolved", port=8802, autostart=True
                ),
            }
        )
        missing_identity = ValueError("local identity unavailable")
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            AsyncMock(side_effect=missing_identity),
        )

        mapping = await manager.local_agent_configs_by_did(config)

        assert mapping == {"did:pkh:warm": ("Warm", config.agents["Warm"])}
        assert manager.cold_scheduler_identity_failures == [
            ("Unresolved", missing_identity)
        ]
        assert manager.is_scheduler_agent_authorized("did:pkh:warm")
        assert not manager.is_scheduler_agent_authorized("did:pkh:unresolved")

    @pytest.mark.asyncio
    async def test_scheduler_preflight_recovers_wal_for_autostart_identity(
        self, monkeypatch, tmp_path,
    ):
        """Autostart authority uses normal WAL recovery before scheduler boot."""

        manager = AgentManager(base_data_dir=tmp_path)
        config = MultiAgentConfig(
            agents={
                "Recovering": LocalAgentConfig(
                    data_dir="recovering", port=8801, autostart=True
                ),
            }
        )
        did_lookup = AsyncMock(return_value="did:pkh:recovered")
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            did_lookup,
        )

        mapping = await manager.local_agent_configs_by_did(config)

        assert mapping == {
            "did:pkh:recovered": ("Recovering", config.agents["Recovering"])
        }
        assert manager.scheduler_authority_for("did:pkh:recovered") == (
            "Recovering",
            config.agents["Recovering"],
        )
        did_lookup.assert_awaited_once_with(
            str(tmp_path / "recovering"),
            mode=AgentDIDLookupMode.INITIALIZATION,
        )

    @pytest.mark.asyncio
    async def test_autostart_preflight_authority_needs_no_runtime_hook(self):
        """A recovered startup DID is already protocol-seeded before load."""

        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        config = LocalAgentConfig(data_dir="recovering", port=8801, autostart=True)
        agent_id = "did:pkh:recovered"
        manager._seed_scheduler_authority({agent_id: ("Recovering", config)})

        assert (
            await manager._begin_dynamic_scheduler_tenant_registration(
                "Recovering", agent_id, config
            )
            is None
        )
        assert not manager.scheduler_lifecycle_lock(agent_id).locked()

    @pytest.mark.asyncio
    async def test_dynamic_scheduler_registration_cancellation_joins_and_rolls_back(
        self,
    ):
        """Cancellation cannot expose scope or orphan protocol/authority state."""

        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        agent_id = "did:scheduler:dynamic-cancel"
        config = LocalAgentConfig(data_dir="dynamic", port=8801)
        hook_started = asyncio.Event()
        release_hook = asyncio.Event()
        protocol_rolled_back = asyncio.Event()

        async def register(_name, _agent_id, _config):
            hook_started.set()
            await release_hook.wait()

            async def rollback():
                protocol_rolled_back.set()

            return rollback

        manager.set_scheduler_tenant_registration_hook(register)
        registration = asyncio.create_task(
            manager._begin_dynamic_scheduler_tenant_registration(
                "Dynamic",
                agent_id,
                config,
            )
        )
        await asyncio.wait_for(hook_started.wait(), timeout=1)
        assert manager.scheduler_authority_for(agent_id) == ("Dynamic", config)
        assert agent_id not in manager.scheduler_authorized_agent_ids()

        registration.cancel()
        await asyncio.sleep(0)
        assert not registration.done()
        release_hook.set()
        with pytest.raises(asyncio.CancelledError):
            await registration

        assert protocol_rolled_back.is_set()
        assert manager.scheduler_authority_for(agent_id) is None
        assert agent_id not in manager.scheduler_authorized_agent_ids()
        assert not manager.scheduler_lifecycle_lock(agent_id).locked()

    @pytest.mark.asyncio
    async def test_dynamic_scheduler_registration_rolls_back_on_onboarding_failure(
        self,
    ):
        """The scheduler lease spans publication and app-owned onboarding."""

        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        agent_id = "did:scheduler:onboarding-failure"
        config = LocalAgentConfig(data_dir="dynamic", port=8801)
        protocol_rolled_back = asyncio.Event()

        async def register(_name, _agent_id, _config):
            async def rollback():
                protocol_rolled_back.set()

            return rollback

        manager.set_scheduler_tenant_registration_hook(register)
        pending = await manager._begin_dynamic_scheduler_tenant_registration(
            "Dynamic",
            agent_id,
            config,
        )
        agent = _make_mock_agent(agent_id)
        agent._dynamic_scheduler_tenant_registration = pending
        manager._initialize_agent = AsyncMock(return_value=agent)
        manager.set_agent_registration_hook(
            AsyncMock(side_effect=RuntimeError("onboarding failed"))
        )

        with pytest.raises(RuntimeError, match="onboarding failed"):
            await manager.load_agent("Dynamic", config)

        assert manager.list_agents() == {}
        assert protocol_rolled_back.is_set()
        assert manager.scheduler_authority_for(agent_id) is None
        assert agent_id not in manager.scheduler_authorized_agent_ids()
        assert not manager.scheduler_lifecycle_lock(agent_id).locked()
        agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cancel_onboarding", [False, True])
    async def test_rejected_onboarding_withdraws_before_a2a_reader_or_cleanup(
        self, cancel_onboarding: bool
    ) -> None:
        """A failed/cancelled hook cannot expose its partial publication."""

        manager = AgentManager()
        agent = _make_mock_agent("did:test:rejected-onboarding")
        config = LocalAgentConfig(data_dir="rejected", port=8801)
        hook_entered = asyncio.Event()
        allow_failure = asyncio.Event()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        async def reject_onboarding(_name, registered_agent) -> None:
            manager.install_a2a_hosted_policy(
                registered_agent,
                resolver=object(),
                authorizer=object(),
                router=object(),
                requester=object(),
            )
            hook_entered.set()
            await allow_failure.wait()
            raise RuntimeError("host onboarding rejected")

        original_cleanup = manager._discard_unpublished_initialized_agents

        async def pause_cleanup(*args, **kwargs) -> bool:
            cleanup_started.set()
            await allow_cleanup.wait()
            return await original_cleanup(*args, **kwargs)

        manager._initialize_agent = AsyncMock(return_value=agent)
        manager.set_agent_registration_hook(reject_onboarding)
        manager._discard_unpublished_initialized_agents = pause_cleanup
        load = asyncio.create_task(manager.load_agent("Rejected", config))
        await asyncio.wait_for(hook_entered.wait(), timeout=1.0)

        async def observe_reader():
            async with manager.a2a_execution_lease():
                return (
                    manager.get_agent("Rejected"),
                    manager.a2a_hosted_policy_for(agent),
                )

        # Queue the reader while onboarding owns the writer.  It can proceed
        # only after that writer releases, exactly where the old failure path
        # exposed routing while it had merely started slow cleanup.
        reader = asyncio.create_task(observe_reader())
        await asyncio.sleep(0)
        if cancel_onboarding:
            load.cancel()
        else:
            allow_failure.set()

        try:
            await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
            observed_agent, observed_policy = await asyncio.wait_for(reader, timeout=1.0)
            assert observed_agent is None
            assert observed_policy is None
        finally:
            allow_cleanup.set()

        if cancel_onboarding:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(load, timeout=1.0)
        else:
            with pytest.raises(RuntimeError, match="host onboarding rejected"):
                await asyncio.wait_for(load, timeout=1.0)
        agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_batch_rejected_onboarding_withdraws_before_a2a_reader_or_cleanup(
        self,
    ) -> None:
        """The ordered batch registrar has the same no-reader failure boundary."""

        manager = AgentManager()
        agent = _make_mock_agent("did:test:batch-rejected-onboarding")
        config = LocalAgentConfig(data_dir="batch-rejected", port=8801)
        hook_entered = asyncio.Event()
        allow_failure = asyncio.Event()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        async def reject_onboarding(_name, registered_agent) -> None:
            manager.install_a2a_hosted_policy(
                registered_agent,
                resolver=object(),
                authorizer=object(),
                router=object(),
                requester=object(),
            )
            hook_entered.set()
            await allow_failure.wait()
            raise RuntimeError("batch host onboarding rejected")

        original_cleanup = manager._discard_unpublished_initialized_agents

        async def pause_cleanup(*args, **kwargs) -> bool:
            cleanup_started.set()
            await allow_cleanup.wait()
            return await original_cleanup(*args, **kwargs)

        manager._initialize_agent = AsyncMock(return_value=agent)
        manager.set_agent_registration_hook(reject_onboarding)
        manager._discard_unpublished_initialized_agents = pause_cleanup
        batch = asyncio.create_task(
            manager.load_from_config(MultiAgentConfig(agents={"Rejected": config}))
        )
        await asyncio.wait_for(hook_entered.wait(), timeout=1.0)

        async def observe_reader():
            async with manager.a2a_execution_lease():
                return (
                    manager.get_agent("Rejected"),
                    manager.a2a_hosted_policy_for(agent),
                )

        reader = asyncio.create_task(observe_reader())
        await asyncio.sleep(0)
        allow_failure.set()
        try:
            await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
            assert await asyncio.wait_for(reader, timeout=1.0) == (None, None)
        finally:
            allow_cleanup.set()

        assert await asyncio.wait_for(batch, timeout=1.0) == 0
        assert manager.init_failures[0][0] == "Rejected"
        agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_agent_cancellation_waiting_for_a2a_publication_cleans_unpublished_dynamic_registration(
        self,
    ):
        """A cancelled cold publication releases its invisible scheduler tenant."""

        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        agent_id = "did:scheduler:cancelled-single-publication"
        config = LocalAgentConfig(data_dir="dynamic", port=8801)
        protocol_rolled_back = asyncio.Event()

        async def register(_name, _agent_id, _config):
            async def rollback() -> None:
                protocol_rolled_back.set()

            return rollback

        manager.set_scheduler_tenant_registration_hook(register)
        pending = await manager._begin_dynamic_scheduler_tenant_registration(
            "Dynamic", agent_id, config
        )
        assert pending is not None
        agent = _make_mock_agent(agent_id)
        agent._dynamic_scheduler_tenant_registration = pending
        initialize_entered = asyncio.Event()
        allow_initialize_return = asyncio.Event()

        async def initialize(*_args, **_kwargs):
            initialize_entered.set()
            await allow_initialize_return.wait()
            return agent

        manager._initialize_agent = initialize
        load = asyncio.create_task(manager.load_agent("Dynamic", config))
        await asyncio.wait_for(initialize_entered.wait(), timeout=1.0)
        await manager._a2a_lifecycle_lock.acquire()
        try:
            allow_initialize_return.set()
            await asyncio.sleep(0)
            assert not load.done()

            load.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(load, timeout=1.0)
        finally:
            manager._a2a_lifecycle_lock.release()

        assert manager.list_agents() == {}
        assert manager.scheduler_authority_for(agent_id) is None
        assert agent_id not in manager.scheduler_authorized_agent_ids()
        assert not manager.scheduler_lifecycle_lock(agent_id).locked()
        assert protocol_rolled_back.is_set()
        agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_from_config_cancellation_waiting_for_a2a_publication_cleans_every_unpublished_result(
        self,
    ):
        """One blocked batch publication cannot strand any initialized sibling."""

        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        configs = {
            "Alpha": LocalAgentConfig(data_dir="alpha", port=8801),
            "Beta": LocalAgentConfig(data_dir="beta", port=8802),
        }
        agent_ids = {
            "Alpha": "did:scheduler:cancelled-batch-alpha",
            "Beta": "did:scheduler:cancelled-batch-beta",
        }
        rolled_back: set[str] = set()

        async def register(name, _agent_id, _config):
            async def rollback() -> None:
                rolled_back.add(name)

            return rollback

        manager.set_scheduler_tenant_registration_hook(register)
        agents = {}
        for name, config in configs.items():
            pending = await manager._begin_dynamic_scheduler_tenant_registration(
                name, agent_ids[name], config
            )
            assert pending is not None
            agent = _make_mock_agent(agent_ids[name])
            agent._dynamic_scheduler_tenant_registration = pending
            agents[name] = agent

        initialized = set()
        all_initialized = asyncio.Event()
        allow_initialize_return = asyncio.Event()

        async def initialize(name, *_args, **_kwargs):
            initialized.add(name)
            if len(initialized) == len(agents):
                all_initialized.set()
            await allow_initialize_return.wait()
            return agents[name]

        manager._initialize_agent = initialize
        batch = asyncio.create_task(
            manager.load_from_config(MultiAgentConfig(agents=configs))
        )
        await asyncio.wait_for(all_initialized.wait(), timeout=1.0)
        await manager._a2a_lifecycle_lock.acquire()
        try:
            allow_initialize_return.set()
            await asyncio.sleep(0)
            assert not batch.done()

            batch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(batch, timeout=1.0)
        finally:
            manager._a2a_lifecycle_lock.release()

        assert manager.list_agents() == {}
        assert rolled_back == {"Alpha", "Beta"}
        for name, agent in agents.items():
            assert manager.scheduler_authority_for(agent_ids[name]) is None
            assert agent_ids[name] not in manager.scheduler_authorized_agent_ids()
            assert not manager.scheduler_lifecycle_lock(agent_ids[name]).locked()
            agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_from_config_cancellation_releases_every_batch_admission(
        self,
    ) -> None:
        """Cancellation during the first final release cannot strand its peers."""

        manager = AgentManager()
        configs = {
            "Alpha": LocalAgentConfig(data_dir="alpha", port=8801),
            "Beta": LocalAgentConfig(data_dir="beta", port=8802),
        }
        agents = {
            "Alpha": _make_mock_agent("did:test:batch-release-alpha"),
            "Beta": _make_mock_agent("did:test:batch-release-beta"),
        }
        final_publication_holds_state_lock = asyncio.Event()

        async def initialize(name, *_args, **_kwargs):
            return agents[name]

        async def hold_state_lock_after_final_publication(name, _agent) -> None:
            if name == "Beta":
                await manager._lock.acquire()
                final_publication_holds_state_lock.set()

        manager._initialize_agent = initialize
        manager.set_agent_registration_hook(hold_state_lock_after_final_publication)
        batch = asyncio.create_task(
            manager.load_from_config(MultiAgentConfig(agents=configs))
        )
        await asyncio.wait_for(final_publication_holds_state_lock.wait(), timeout=1.0)
        try:
            # The batch has committed both agents and is blocked retiring its
            # first admission. Its cancellation must join both release tasks.
            batch.cancel()
            await asyncio.sleep(0)
        finally:
            manager._lock.release()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(batch, timeout=1.0)

        assert manager.list_agents() == agents
        assert manager._agent_operations == {}

    @pytest.mark.asyncio
    async def test_committed_load_returns_success_after_cancellation_during_admission_release(
        self,
    ) -> None:
        """A completed load is success even when its final release sees cancel."""

        manager = AgentManager()
        agent = _make_mock_agent("did:test:committed-load")
        release_is_blocked = asyncio.Event()

        async def hold_state_lock_after_registration(_name, _agent) -> None:
            await manager._lock.acquire()
            release_is_blocked.set()

        manager._initialize_agent = AsyncMock(return_value=agent)
        manager.set_agent_registration_hook(hold_state_lock_after_registration)
        load = asyncio.create_task(
            manager.load_agent(
                "CommittedLoad", LocalAgentConfig(data_dir="load", port=8801)
            )
        )
        await asyncio.wait_for(release_is_blocked.wait(), timeout=1.0)
        try:
            load.cancel()
            await asyncio.sleep(0)
        finally:
            manager._lock.release()

        assert await asyncio.wait_for(load, timeout=1.0) is agent
        assert manager.get_agent("CommittedLoad") is agent
        assert manager._agent_operations == {}

    @pytest.mark.asyncio
    async def test_committed_create_returns_success_after_cancellation_during_admission_release(
        self, monkeypatch, tmp_path
    ) -> None:
        """A committed create preserves its persistence handoff after cancel."""

        manager = AgentManager(base_data_dir=tmp_path)
        agent = _make_mock_agent("did:test:committed-create")
        release_is_blocked = asyncio.Event()

        async def hold_state_lock_after_registration(_name, _agent) -> None:
            await manager._lock.acquire()
            release_is_blocked.set()

        manager._data_key_custody_conflict = lambda: None
        manager._initialize_agent = AsyncMock(return_value=agent)
        manager.set_agent_registration_hook(hold_state_lock_after_registration)
        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            AsyncMock(),
        )
        create = asyncio.create_task(manager.create_agent("CommittedCreate"))
        await asyncio.wait_for(release_is_blocked.wait(), timeout=1.0)
        try:
            create.cancel()
            await asyncio.sleep(0)
        finally:
            manager._lock.release()

        assert await asyncio.wait_for(create, timeout=1.0) is agent
        assert manager.get_agent("CommittedCreate") is agent
        assert "CommittedCreate" in manager._created_configs
        assert manager._agent_operations == {}

    @pytest.mark.asyncio
    async def test_pending_dynamic_registration_cannot_claim_before_onboarding_commit(
        self, tmp_path
    ):
        """A host runner cannot adopt rollback-owned rows before onboarding."""

        from kestrel_sovereign.features.scheduler.runner import (
            SCHEDULER_PROTOCOL_VERSION,
            SchedulerRunner,
        )
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.db import SQLiteBackend

        backend = SQLiteBackend(str(tmp_path / "pending-registration.db"))
        await backend.connect()
        db = AsyncDatabase(backend)
        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        agent_id = "did:scheduler:pending-onboarding"
        config = LocalAgentConfig(data_dir="dynamic", port=8801)
        registration_runner = SchedulerRunner(
            db,
            None,
            AsyncMock(),
            authorized_agent_ids=(agent_id,),
            owner_id="registration-owner",
        )
        pending = None
        try:
            async def register(_name, _agent_id, _config):
                durable_registration = (
                    await registration_runner.prepare_tenant_registration()
                )

                async def rollback() -> None:
                    await registration_runner.rollback_tenant_registration(
                        durable_registration
                    )

                rollback.scheduler_registration_nonce = (
                    durable_registration.registration_nonce
                )
                return rollback

            manager.set_scheduler_tenant_registration_hook(register)
            pending = await manager._begin_dynamic_scheduler_tenant_registration(
                "Dynamic", agent_id, config
            )
            assert pending is not None
            assert manager.scheduler_authority_for(agent_id) == ("Dynamic", config)
            assert not manager.is_scheduler_agent_authorized(agent_id)

            now = datetime.now(timezone.utc).isoformat()
            due = "2000-01-01T00:00:00+00:00"
            await db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, args_json, enabled,
                     next_run_at, created_at, scheduler_protocol_version,
                     scheduler_registration_nonce)
                VALUES (?, ?, 'backup_snapshot', '* * * * *', '{}', 1, ?, ?, ?, ?)
                """,
                (
                    "pending-owned-schedule",
                    agent_id,
                    due,
                    now,
                    SCHEDULER_PROTOCOL_VERSION,
                    pending.registration_nonce,
                ),
            )
            host_runner = SchedulerRunner(
                db,
                None,
                AsyncMock(return_value="executed"),
                authorized_agent_ids=(agent_id,),
                authorized_agent_ids_provider=manager.scheduler_authorized_agent_ids,
                is_agent_authorized=manager.is_scheduler_agent_authorized,
                owner_id="host-runner",
            )

            # This is the former race: a scope publication here let the host
            # claim the row, clear its nonce, and make rollback retain it.
            await host_runner._tick()
            assert await db.fetchone(
                "SELECT scheduler_registration_nonce FROM scheduled_tasks WHERE id = ?",
                ("pending-owned-schedule",),
            ) == (pending.registration_nonce,)
            assert await db.fetchone(
                "SELECT COUNT(*) FROM task_execution_log WHERE task_id = ?",
                ("pending-owned-schedule",),
            ) == (0,)

            await pending.rollback()
            pending = None
            assert await db.fetchone(
                "SELECT COUNT(*) FROM scheduled_tasks WHERE id = ?",
                ("pending-owned-schedule",),
            ) == (0,)
            assert await db.fetchone(
                "SELECT COUNT(*) FROM task_execution_log WHERE task_id = ?",
                ("pending-owned-schedule",),
            ) == (0,)
        finally:
            if pending is not None:
                await pending.rollback()
            await db.close()

    @pytest.mark.asyncio
    async def test_hosted_cold_registration_validates_authority_without_releasing_owned_lock(
        self,
    ):
        """The explicit lock-owner path never reacquires or releases the DID lease."""

        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        agent_id = "did:scheduler:configured-cold"
        config = LocalAgentConfig(data_dir="cold", port=8801, autostart=False)
        registration_hook = AsyncMock()
        manager.set_scheduler_tenant_registration_hook(registration_hook)
        lifecycle_lock = manager.scheduler_lifecycle_lock(agent_id)
        await lifecycle_lock.acquire()
        try:
            with pytest.raises(LookupError, match="without live manager authority"):
                await manager._begin_dynamic_scheduler_tenant_registration(
                    "Cold",
                    agent_id,
                    config,
                    scheduler_lifecycle_lock_held=True,
                )
            assert lifecycle_lock.locked()

            manager._seed_scheduler_authority(
                {agent_id: ("Cold", config)}
            )
            assert (
                await manager._begin_dynamic_scheduler_tenant_registration(
                    "Cold",
                    agent_id,
                    config,
                    scheduler_lifecycle_lock_held=True,
                )
                is None
            )
            assert lifecycle_lock.locked()
            registration_hook.assert_not_awaited()

            manager._scheduler_authority_by_name["Cold"] = "did:other"
            with pytest.raises(RuntimeError, match="does not match"):
                await manager._begin_dynamic_scheduler_tenant_registration(
                    "Cold",
                    agent_id,
                    config,
                    scheduler_lifecycle_lock_held=True,
                )
            assert lifecycle_lock.locked()
        finally:
            lifecycle_lock.release()

    @pytest.mark.asyncio
    async def test_cold_scheduler_load_refuses_did_mismatch_before_registration(
        self, tmp_path,
    ):
        """A mismatched cold wake releases every private authority reservation."""
        manager = AgentManager(base_data_dir=tmp_path)
        parent, mandate = _signed_restored_mandate(
            "did:pkh:mismatch-parent",
            "did:pkh:tenant-b",
        )
        tenant_b_agent = _make_mock_agent(mandate.child_did)
        tenant_b_agent._persisted_spawn_mandate = mandate
        manager._register_agent("MismatchParent", parent)

        async def initialize_with_reserved_authority(name, _config, **_kwargs):
            manager._initializing_agents[name] = tenant_b_agent
            manager._verify_agent_authority(name, tenant_b_agent)
            assert manager._preflight_spawn_reservations
            return tenant_b_agent

        manager._initialize_agent = AsyncMock(
            side_effect=initialize_with_reserved_authority
        )
        config = LocalAgentConfig(
            data_dir="cold", port=8801, autostart=False
        )
        manager._seed_scheduler_authority(
            {"did:pkh:tenant-a": ("Cold", config)}
        )

        with pytest.raises(RuntimeError, match="does not match claimed DID"):
            await manager.load_agent(
                "Cold",
                config,
                expected_agent_id="did:pkh:tenant-a",
            )

        assert manager.list_agents() == {"MismatchParent": parent}
        assert manager.get_agent_name("did:pkh:tenant-b") is None
        assert manager._preflight_spawn_reservations == {}
        assert tenant_b_agent._agent_manager_authority_reserved is None
        tenant_b_agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remove_agent(self):
        manager = AgentManager()
        mock = _make_mock_agent("did:pkh:remove")
        manager._agents["Testbot"] = mock
        manager._agent_names["did:pkh:remove"] = "Testbot"

        result = await manager.remove_agent("Testbot")
        assert result is True
        assert manager.get_agent("Testbot") is None
        assert manager.get_agent_name("did:pkh:remove") is None
        mock.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remove_agent_preserves_refund_failure_with_cancellation(self):
        """A cancellation outcome cannot erase a settled refund failure."""

        manager = AgentManager()
        child = _make_mock_agent("did:test:cancelled-refund-failure")
        manager._agents["Child"] = child
        manager._agent_names[child.agent_id] = "Child"
        budget_entry = (object(), object())
        manager._child_budgets["Child"] = budget_entry
        refund_failure = RuntimeError("refund provider unavailable")

        async def fail_refund(name: str) -> bool:
            assert name == "Child"
            assert manager._child_budgets[name] is budget_entry
            raise refund_failure

        manager._release_child_budget_cancellation_safe = fail_refund
        manager._reconcile_fully_removed_child_tracking = AsyncMock(
            return_value=True
        )

        with pytest.raises(BaseExceptionGroup) as raised:
            await manager.remove_agent("Child")

        assert str(raised.value).startswith(
            "Agent 'Child' removal had terminal budget outcomes"
        )
        assert len(raised.value.exceptions) == 2
        assert isinstance(raised.value.exceptions[0], asyncio.CancelledError)
        assert raised.value.exceptions[1] is refund_failure
        assert manager.get_agent("Child") is None
        assert manager._child_budgets["Child"] is budget_entry

    @pytest.mark.asyncio
    async def test_remove_agent_securely_deletes_only_owned_hosted_runtime_tree(
        self, tmp_path
    ):
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:hosted-offboarding"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(
            scope,
            did,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        (scope.path / "feature_venvs" / "feature-safe" / "credential").write_text(
            "tenant-secret"
        )
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep").write_text("unrelated")
        (scope.path / "outside-link").symlink_to(outside, target_is_directory=True)
        agent = _make_mock_agent(did)
        agent.did = did
        agent.isolated_runtime_scope = scope
        manager._agents["Hosted"] = agent
        manager._agent_names[did] = "Hosted"

        assert (
            await manager.remove_agent(
                "Hosted",
                offboard_runtime=True,
            )
            is True
        )

        assert not scope.path.exists()
        assert scope.root.is_dir()
        assert (outside / "keep").read_text() == "unrelated"

    @pytest.mark.asyncio
    async def test_offboard_removes_unadopted_released_feature_credentials(
        self,
        tmp_path,
    ):
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:released-offboarding"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(scope, did)
        legacy_root = tmp_path / "agent_data" / "hosted" / "feature_venvs"
        legacy_feature = legacy_root / "NeverLoadedWhatsAppFeature"
        legacy_feature.mkdir(parents=True)
        legacy_root.chmod(0o700)
        legacy_feature.chmod(0o700)
        credential = legacy_feature / "auth" / "credentials.json"
        credential.parent.mkdir()
        credential.write_text("legacy-tenant-secret")
        agent = _make_mock_agent(did)
        agent.did = did
        agent.isolated_runtime_scope = scope
        agent.isolated_runtime_legacy_root = legacy_root
        manager._agents["Hosted"] = agent
        manager._agent_names[did] = "Hosted"

        assert await manager.remove_agent("Hosted", offboard_runtime=True)

        assert not scope.path.exists()
        assert not legacy_root.exists()

    @pytest.mark.asyncio
    async def test_offboard_never_reports_removed_while_legacy_custody_remains(
        self,
        tmp_path,
    ):
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:ambiguous-released-offboarding"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(scope, did)
        legacy_root = tmp_path / "agent_data" / "hosted" / "feature_venvs"
        legacy_feature = legacy_root / "NeverLoadedWhatsAppFeature"
        legacy_feature.mkdir(parents=True)
        legacy_root.chmod(0o775)
        credential = legacy_feature / "credentials.json"
        credential.write_text("ambiguous-custody")
        agent = _make_mock_agent(did)
        agent.did = did
        agent.isolated_runtime_scope = scope
        agent.isolated_runtime_legacy_root = legacy_root
        manager._agents["Hosted"] = agent
        manager._agent_names[did] = "Hosted"

        with pytest.raises(RuntimeOffboardingRetainedError) as raised:
            await manager.remove_agent("Hosted", offboard_runtime=True)

        assert raised.value.metadata["runtime_cleanup_state"] == "retained"
        assert manager.get_agent("Hosted") is None
        assert credential.read_text() == "ambiguous-custody"

    @pytest.mark.asyncio
    async def test_unpublished_budget_hold_still_admits_runtime_offboarding(
        self,
        monkeypatch,
        tmp_path,
    ):
        """A late hold cannot turn destructive rollback into refund-only success."""

        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://host/kestrel")
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:late-budget-hold"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(scope, did)
        credential = scope.path / "credential"
        credential.write_text("must-be-offboarded")
        config = LocalAgentConfig(data_dir="agent_data/Kid", port=8801)
        config.resolve_data_dir(tmp_path).mkdir(parents=True)
        manager._created_configs["Kid"] = config
        delegated = SimpleNamespace(
            allocation=SimpleNamespace(child_did=did),
        )
        manager._child_budgets["Kid"] = (delegated, object())

        async def release_late_hold(child_name: str) -> None:
            assert child_name == "Kid"
            manager._child_budgets.pop(child_name)

        manager._release_child_budget_cancellation_safe = release_late_hold

        assert await manager.remove_agent("Kid", offboard_runtime=True)

        assert not scope.path.exists()
        assert "Kid" not in manager._child_budgets

    @pytest.mark.asyncio
    async def test_cold_budgeted_descendant_blocks_identity_offboarding(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://host/kestrel")
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:cold-parent-with-budget"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(scope, did)
        manager._parent_children[did] = ["Grandchild"]
        manager._child_budgets["Grandchild"] = (object(), object())

        with pytest.raises(ValueError, match="budgeted child agents"):
            await manager.remove_agent(
                "ColdParent",
                offboard_runtime=True,
                known_agent_id=did,
            )

        assert scope.path.exists()
        assert "Grandchild" in manager._child_budgets

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("hosted_scope", "cleanup_state", "runtime_retained"),
        (
            (False, "not_hosted", True),
            (True, "already_absent", False),
        ),
        ids=("storage-backed", "already-absent"),
    )
    async def test_destructive_remove_reports_when_no_runtime_tree_was_deleted(
        self,
        tmp_path,
        hosted_scope,
        cleanup_state,
        runtime_retained,
    ):
        manager = AgentManager(base_data_dir=tmp_path)
        did = f"did:pkh:no-delete-{cleanup_state}"
        agent = _make_mock_agent(did)
        agent.did = did
        if hosted_scope:
            agent.isolated_runtime_scope = resolve_isolated_runtime_namespace(
                manager._isolated_runtime_root,
                derive_isolated_runtime_namespace(did),
            )
            assert not agent.isolated_runtime_scope.path.exists()
        manager._agents["Hosted"] = agent
        manager._agent_names[did] = "Hosted"

        with pytest.raises(RuntimeOffboardingNotPerformedError) as raised:
            await manager.remove_agent("Hosted", offboard_runtime=True)

        assert agent.shutdown.await_count == 1
        assert manager.get_agent("Hosted") is None
        assert raised.value.metadata == {
            "code": "runtime_offboarding_not_performed",
            "agent": "Hosted",
            "agent_id": did,
            "agent_removed": True,
            "runtime_offboard_requested": True,
            "runtime_offboarded": False,
            "runtime_retained": runtime_retained,
            "runtime_cleanup_pending": False,
            "runtime_cleanup_state": cleanup_state,
            "runtime_already_absent": hosted_scope,
            "hosted_runtime_configured": hosted_scope,
        }

    @pytest.mark.asyncio
    async def test_slow_runtime_offboarding_is_bounded_outside_lifecycle_locks(
        self, monkeypatch
    ):
        manager = AgentManager()
        hosted = _make_mock_agent("did:pkh:slow-offboard")
        other = _make_mock_agent("did:pkh:unrelated")
        manager._agents.update({"Hosted": hosted, "Other": other})
        manager._agent_names.update(
            {hosted.agent_id: "Hosted", other.agent_id: "Other"}
        )
        started = threading.Event()
        release = threading.Event()

        def slow_cleanup(_agent):
            started.set()
            release.wait(timeout=5)
            return RuntimeNamespaceCleanupOutcome.REMOVED

        monkeypatch.setattr(
            "kestrel_sovereign.features.isolated_runtime.remove_agent_runtime_namespace",
            slow_cleanup,
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.RUNTIME_OFFBOARD_TIMEOUT_S",
            0.05,
        )

        deletion = asyncio.create_task(
            manager.remove_agent("Hosted", offboard_runtime=True)
        )
        try:
            assert await asyncio.to_thread(started.wait, 1)
            # The slow filesystem worker cannot retain the manager-wide or
            # A2A lifecycle locks needed by an unrelated agent removal.
            assert await asyncio.wait_for(manager.remove_agent("Other"), 0.5)
            with pytest.raises(RuntimeOffboardingRetainedError) as raised:
                await deletion
            assert raised.value.metadata["cause_type"] == "TimeoutError"
            assert raised.value.metadata["runtime_cleanup_pending"] is True
            assert raised.value.metadata["runtime_cleanup_state"] == "pending"
            assert "still pending" in str(raised.value)
            assert "was retained" not in str(raised.value)
            assert "runtime_path" not in raised.value.metadata
            assert manager.get_agent("Hosted") is None
            assert manager.quarantined_shutdowns()
        finally:
            release.set()
        await manager.drain_quarantined_shutdowns()
        completed = list(manager.quarantined_shutdowns().values())
        assert completed
        assert all(record["pending"] is False for record in completed)
        assert all(record["failure"] is None for record in completed)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("outcome", "failure_type"),
        (
            (
                RuntimeNamespaceCleanupOutcome.ALREADY_ABSENT,
                "RuntimeOffboardingNotPerformedError",
            ),
            (None, "TypeError"),
        ),
        ids=("already-absent", "invalid-none"),
    )
    async def test_deferred_offboarding_records_non_removed_outcome_as_unsafe(
        self,
        monkeypatch,
        outcome,
        failure_type,
    ):
        """A timed-out worker's eventual return must resolve pending custody."""

        manager = AgentManager()
        agent = _make_mock_agent("did:pkh:deferred-non-removal")
        manager._agents["Hosted"] = agent
        manager._agent_names[agent.agent_id] = "Hosted"
        started = threading.Event()
        release = threading.Event()

        def delayed_outcome(_agent):
            started.set()
            release.wait(timeout=5)
            return outcome

        monkeypatch.setattr(
            "kestrel_sovereign.features.isolated_runtime.remove_agent_runtime_namespace",
            delayed_outcome,
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.RUNTIME_OFFBOARD_TIMEOUT_S",
            0.01,
        )
        removal = asyncio.create_task(
            manager.remove_agent("Hosted", offboard_runtime=True)
        )
        try:
            assert await asyncio.to_thread(started.wait, 1)
            with pytest.raises(RuntimeOffboardingRetainedError):
                await removal
        finally:
            release.set()

        with pytest.raises(ExceptionGroup, match="quarantined shutdown reapers"):
            await manager.drain_quarantined_shutdowns()

        completed = list(manager.quarantined_shutdowns().values())
        assert len(completed) == 1
        assert completed[0]["pending"] is False
        assert failure_type in completed[0]["failure"]
        assert manager.get_agent("Hosted") is None

    @pytest.mark.asyncio
    async def test_cancelled_runtime_offboarding_is_retained_and_does_not_wedge_peers(
        self, monkeypatch
    ):
        manager = AgentManager()
        hosted = _make_mock_agent("did:pkh:cancelled-cleanup")
        other = _make_mock_agent("did:pkh:cancelled-peer")
        manager._agents.update({"Hosted": hosted, "Other": other})
        manager._agent_names.update(
            {hosted.agent_id: "Hosted", other.agent_id: "Other"}
        )
        started = threading.Event()
        release = threading.Event()

        def slow_cleanup(_agent):
            started.set()
            release.wait(timeout=5)
            return RuntimeNamespaceCleanupOutcome.REMOVED

        monkeypatch.setattr(
            "kestrel_sovereign.features.isolated_runtime.remove_agent_runtime_namespace",
            slow_cleanup,
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.RUNTIME_OFFBOARD_TIMEOUT_S",
            5.0,
        )
        deletion = asyncio.create_task(
            manager.remove_agent("Hosted", offboard_runtime=True)
        )
        try:
            assert await asyncio.to_thread(started.wait, 1)
            deletion.cancel()
            with pytest.raises(BaseExceptionGroup) as raised:
                await deletion
            assert _exception_group_contains(raised.value, asyncio.CancelledError)
            assert _exception_group_contains(
                raised.value, RuntimeOffboardingRetainedError
            )
            retained = next(
                error
                for error in _exception_leaves(raised.value)
                if isinstance(error, RuntimeOffboardingRetainedError)
            )
            assert retained.metadata["runtime_cleanup_pending"] is True
            assert retained.metadata["runtime_cleanup_state"] == "pending"
            assert manager.get_agent("Hosted") is None
            assert await asyncio.wait_for(manager.remove_agent("Other"), 0.5)
        finally:
            release.set()
        await manager.drain_quarantined_shutdowns()

    @pytest.mark.asyncio
    async def test_shutdown_all_preserves_hosted_runtime_tree(self, tmp_path):
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:hosted-restart"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(
            scope,
            did,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        credential = scope.path / "feature_venvs" / "feature-safe" / "credential"
        credential.write_text("survive-restart")
        agent = _make_mock_agent(did)
        agent.did = did
        agent.isolated_runtime_scope = scope
        manager._agents["Hosted"] = agent
        manager._agent_names[did] = "Hosted"

        await manager.shutdown_all()

        assert manager.get_agent("Hosted") is None
        assert credential.read_text() == "survive-restart"
        assert (scope.path / ".kestrel-runtime-owner").is_file()

    @pytest.mark.asyncio
    async def test_delete_endpoint_defaults_to_state_preserving_stop(self):
        from kestrel_sovereign.endpoints.models import delete_agent

        manager = SimpleNamespace(remove_agent=AsyncMock(return_value=True))
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(agent_manager=manager))
        )

        result = await delete_agent.__wrapped__(request, "Hosted")

        manager.remove_agent.assert_awaited_once_with(
            "Hosted",
            offboard_runtime=False,
        )
        assert result["success"] is True
        assert result["runtime_offboarded"] is False
        assert result["runtime_retained_for_restart"] is True

    @pytest.mark.asyncio
    async def test_delete_endpoint_default_does_not_rewrite_autostart_config(
        self,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        config = MultiAgentConfig(
            agents={
                "Hosted": LocalAgentConfig(
                    data_dir=Path("agent_data/hosted"),
                    port=8801,
                    autostart=True,
                )
            }
        )
        config.save(config_path)
        before = config_path.read_bytes()
        manager = SimpleNamespace(remove_agent=AsyncMock(return_value=True))
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        result = await delete_agent.__wrapped__(request, "Hosted")

        assert config_path.read_bytes() == before
        assert "Hosted" in MultiAgentConfig.from_file(config_path).agents
        assert result["runtime_retained_for_restart"] is True

    @pytest.mark.asyncio
    async def test_delete_endpoint_explicit_offboard_removes_autostart_registration(
        self,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        config = MultiAgentConfig(
            agents={
                "Hosted": LocalAgentConfig(
                    data_dir=Path("agent_data/hosted"),
                    port=8801,
                    autostart=True,
                )
            }
        )
        config.save(config_path)
        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(return_value="did:test:hosted"),
            remove_agent=_admitted_offboarding_success(),
        )
        state = SimpleNamespace(
            agent_manager=manager,
            multi_agent_config_path=config_path,
            multi_agent_config=config,
        )
        request = SimpleNamespace(app=SimpleNamespace(state=state))

        result = await delete_agent.__wrapped__(
            request,
            "Hosted",
            offboard_runtime=True,
        )

        manager.remove_agent.assert_awaited_once_with(
            "Hosted",
            offboard_runtime=True,
            known_agent_id="did:test:hosted",
            known_agent_config=config.agents["Hosted"],
            offboarding_admission=manager.remove_agent.await_args.kwargs[
                "offboarding_admission"
            ],
        )
        admission = manager.remove_agent.await_args.kwargs["offboarding_admission"]
        assert isinstance(admission, RuntimeOffboardingAdmission)
        assert admission.started is True
        assert MultiAgentConfig.from_file(config_path).agents == {}
        assert state.multi_agent_config.agents == {}
        assert result["runtime_offboarded"] is True
        assert result["persisted_registration_removed"] is True

    @pytest.mark.asyncio
    async def test_cold_delete_restores_registration_when_offboarding_not_admitted(
        self,
        monkeypatch,
        tmp_path,
    ):
        """Cold routing absence cannot suppress pre-admission compensation."""

        from kestrel_sovereign.endpoints.models import delete_agent

        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://host/kestrel")
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:test:cold-admission-compensation"
        local = LocalAgentConfig(
            data_dir=Path("agent_data/cold"),
            port=8801,
            autostart=True,
        )
        local.resolve_data_dir(tmp_path).mkdir(parents=True)
        manager._seed_scheduler_authority({did: ("Cold", local)})
        manager.resolve_registered_agent_id = AsyncMock(return_value=did)
        start_failure = OSError("private operator mount")
        start = MagicMock(side_effect=start_failure)
        monkeypatch.setattr(
            manager,
            "_start_agent_runtime_offboarding_identity",
            start,
        )
        config = MultiAgentConfig(agents={"Cold": local})
        config_path = tmp_path / "multi_agent.toml"
        config.save(config_path)
        state = SimpleNamespace(
            agent_manager=manager,
            multi_agent_config_path=config_path,
            multi_agent_config=config,
        )
        request = SimpleNamespace(app=SimpleNamespace(state=state))

        with pytest.raises(OSError) as raised:
            await delete_agent.__wrapped__(
                request,
                "Cold",
                offboard_runtime=True,
            )

        assert raised.value is start_failure
        assert "Cold" in MultiAgentConfig.from_file(config_path).agents
        assert "Cold" in state.multi_agent_config.agents
        assert manager.scheduler_authority_for(did) == ("Cold", local)
        assert manager.is_scheduler_agent_authorized(did)
        assert manager.get_agent("Cold") is None
        assert manager._inflight_runtime_offboardings == {}
        start.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_endpoint_refuses_destructive_auto_discovery_deployment(self):
        from kestrel_sovereign.endpoints.models import delete_agent

        manager = SimpleNamespace(remove_agent=AsyncMock(return_value=True))
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=None,
                )
            )
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 409
        assert "auto-discovered" in raised.value.detail
        manager.remove_agent.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("cleanup_state", "runtime_retained"),
        (("not_hosted", True), ("already_absent", False)),
    )
    async def test_delete_endpoint_never_claims_a_noop_runtime_offboard(
        self,
        tmp_path,
        cleanup_state,
        runtime_retained,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        config = MultiAgentConfig(
            agents={
                "Hosted": LocalAgentConfig(
                    data_dir=Path("agent_data/hosted"),
                    port=8801,
                )
            }
        )
        config.save(config_path)
        outcome = RuntimeOffboardingNotPerformedError(
            agent_name="Hosted",
            agent_id="did:test:no-delete",
            cleanup_state=cleanup_state,
        )
        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(return_value="did:test:no-delete"),
            remove_agent=_admitted_offboarding_failure(outcome),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 409
        assert raised.value.detail["runtime_offboarded"] is False
        assert raised.value.detail["runtime_retained"] is runtime_retained
        assert raised.value.detail["runtime_cleanup_state"] == cleanup_state
        assert raised.value.detail["persisted_registration_removed"] is True
        assert MultiAgentConfig.from_file(config_path).agents == {}

    @pytest.mark.asyncio
    async def test_delete_endpoint_retains_config_removal_while_cleanup_is_pending(
        self,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        config = MultiAgentConfig(
            agents={
                "Hosted": LocalAgentConfig(
                    data_dir=Path("agent_data/hosted"),
                    port=8801,
                )
            }
        )
        config.save(config_path)
        pending = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id="did:test:pending",
            runtime_path=tmp_path / "private-runtime",
            cause=TimeoutError("private shutdown handoff"),
            cleanup_pending=True,
        )
        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(return_value="did:test:pending"),
            remove_agent=_admitted_offboarding_failure(pending),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 409
        assert raised.value.detail["runtime_cleanup_state"] == "pending"
        assert MultiAgentConfig.from_file(config_path).agents == {}
        assert str(tmp_path) not in str(raised.value.detail)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("outcome_kind", "expected_code", "expected_cleanup_state"),
        (
            (
                "retained",
                "runtime_offboarding_retained",
                "retained",
            ),
            (
                "not-performed",
                "runtime_offboarding_not_performed",
                "already_absent",
            ),
            (
                "grouped-retained",
                "runtime_offboarding_retained",
                "retained",
            ),
        ),
    )
    async def test_delete_endpoint_preserves_custody_409_on_restore_conflict(
        self,
        tmp_path,
        caplog,
        outcome_kind,
        expected_code,
        expected_cleanup_state,
    ):
        """A concurrent config writer cannot replace typed custody with 500."""

        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        original = LocalAgentConfig(
            data_dir=Path("agent_data/original"),
            port=8801,
            autostart=True,
        )
        replacement = LocalAgentConfig(
            data_dir=Path("agent_data/concurrent-private-registration"),
            port=8899,
            autostart=False,
        )
        config = MultiAgentConfig(agents={"Hosted": original})
        config.save(config_path)
        private_path = tmp_path / "private-runtime" / "credential.json"
        outcome = _endpoint_custody_outcome(
            outcome_kind,
            private_path=private_path,
        )

        async def conflict_then_fail(_name, **kwargs):
            admission = kwargs["offboarding_admission"]
            assert isinstance(admission, RuntimeOffboardingAdmission)
            assert admission.started is False
            # A concurrent writer legitimately wins after the endpoint's CAS
            # removal. Compensation must refuse to overwrite this new row.
            MultiAgentConfig(agents={"Hosted": replacement}).save(config_path)
            raise outcome

        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(
                return_value="did:test:restore-conflict"
            ),
            remove_agent=AsyncMock(side_effect=conflict_then_fail),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )
        caplog.set_level(
            "ERROR",
            logger="kestrel_sovereign.endpoints.models",
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 409
        detail = raised.value.detail
        assert detail["code"] == expected_code
        assert detail["runtime_cleanup_state"] == expected_cleanup_state
        assert detail["persisted_registration_removed"] is True
        assert detail["persisted_registration_requires_reconciliation"] is True
        assert str(private_path) not in str(detail)
        assert "private cleanup failure" not in str(detail)
        assert "private refund failure" not in str(detail)
        assert (
            MultiAgentConfig.from_file(config_path).agents["Hosted"]
            == replacement
        )
        assert "operator reconciliation is required" in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "outcome_kind",
        ("retained", "not-performed", "grouped-retained"),
    )
    async def test_delete_endpoint_custody_409_reports_registration_restored(
        self,
        tmp_path,
        outcome_kind,
    ):
        """A successful pre-admission compensation retains the original row."""

        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        original = LocalAgentConfig(
            data_dir=Path("agent_data/original"),
            port=8801,
            autostart=True,
        )
        config = MultiAgentConfig(agents={"Hosted": original})
        config.save(config_path)
        outcome = _endpoint_custody_outcome(
            outcome_kind,
            private_path=tmp_path / "private-runtime" / "credential.json",
        )
        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(return_value="did:test:hosted"),
            remove_agent=AsyncMock(side_effect=outcome),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 409
        detail = raised.value.detail
        assert detail["persisted_registration_removed"] is False
        assert "persisted_registration_requires_reconciliation" not in detail
        assert MultiAgentConfig.from_file(config_path).agents["Hosted"] == original
        assert request.app.state.multi_agent_config.agents["Hosted"] == original

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("outcome_kind", "expected_code", "expected_cleanup_state"),
        (
            (
                "retained",
                "runtime_offboarding_retained",
                "retained",
            ),
            (
                "not-performed",
                "runtime_offboarding_not_performed",
                "already_absent",
            ),
            (
                "grouped-retained",
                "runtime_offboarding_retained",
                "retained",
            ),
        ),
    )
    @pytest.mark.parametrize(
        "restore_failure",
        ("missing", "invalid", "write-failure"),
    )
    async def test_delete_endpoint_marks_removed_when_config_restore_fails(
        self,
        tmp_path,
        monkeypatch,
        caplog,
        outcome_kind,
        expected_code,
        expected_cleanup_state,
        restore_failure,
    ):
        """Unreadable or unwritable desired state retains truthful custody."""

        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        original = LocalAgentConfig(
            data_dir=Path("agent_data/original"),
            port=8801,
            autostart=True,
        )
        config = MultiAgentConfig(agents={"Hosted": original})
        config.save(config_path)
        private_path = tmp_path / "private-runtime" / "credential.json"
        outcome = _endpoint_custody_outcome(
            outcome_kind,
            private_path=private_path,
        )

        if restore_failure == "write-failure":
            real_save = MultiAgentConfig.save
            save_calls = 0

            def fail_restore_save(candidate, path=None):
                nonlocal save_calls
                save_calls += 1
                if save_calls == 2:
                    raise PermissionError(
                        f"private config volume at {tmp_path / 'operator.toml'}"
                    )
                return real_save(candidate, path)

            monkeypatch.setattr(MultiAgentConfig, "save", fail_restore_save)

        async def damage_config_then_fail(name, **kwargs):
            assert name == "Hosted"
            admission = kwargs["offboarding_admission"]
            assert isinstance(admission, RuntimeOffboardingAdmission)
            assert admission.started is False
            if restore_failure == "missing":
                config_path.unlink()
            elif restore_failure == "invalid":
                config_path.write_text("[agents.invalid", encoding="utf-8")
            raise outcome

        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(
                return_value="did:test:restore-conflict"
            ),
            remove_agent=AsyncMock(side_effect=damage_config_then_fail),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )
        caplog.set_level(
            "ERROR",
            logger="kestrel_sovereign.endpoints.models",
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "hOsTeD",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 409
        detail = raised.value.detail
        assert detail["code"] == expected_code
        assert detail["runtime_cleanup_state"] == expected_cleanup_state
        assert detail["persisted_registration_removed"] is True
        assert detail["persisted_registration_requires_reconciliation"] is True
        assert str(private_path) not in str(detail)
        assert "private cleanup failure" not in str(detail)
        assert "private refund failure" not in str(detail)
        assert "operator.toml" not in str(detail)
        assert "operator reconciliation is required" in caplog.text
        assert "agent 'Hosted'" in caplog.text
        assert "hOsTeD" not in caplog.text
        if restore_failure == "missing":
            assert not config_path.exists()
        elif restore_failure == "invalid":
            with pytest.raises(ValueError, match="Invalid TOML"):
                MultiAgentConfig.from_file(config_path)
        else:
            assert MultiAgentConfig.from_file(config_path).agents == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("manager_outcome", "expected_log"),
        (
            ("value-error", "refused offboarding of 'Hosted'"),
            ("not-found", "after agent 'Hosted' was not found"),
        ),
    )
    async def test_delete_endpoint_reconciliation_logs_use_persisted_key(
        self,
        tmp_path,
        caplog,
        manager_outcome,
        expected_log,
    ):
        """All post-resolution reconciliation alerts use the TOML key."""

        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        original = LocalAgentConfig(
            data_dir=Path("agent_data/original"),
            port=8801,
            autostart=True,
        )
        replacement = LocalAgentConfig(
            data_dir=Path("agent_data/concurrent"),
            port=8899,
            autostart=False,
        )
        config = MultiAgentConfig(agents={"Hosted": original})
        config.save(config_path)

        async def conflict_after_removal(name, **_kwargs):
            assert name == "Hosted"
            MultiAgentConfig(agents={"Hosted": replacement}).save(config_path)
            if manager_outcome == "value-error":
                raise ValueError(f"private refusal at {tmp_path}")
            return False

        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(return_value="did:test:hosted"),
            remove_agent=AsyncMock(side_effect=conflict_after_removal),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )
        caplog.set_level(
            "ERROR",
            logger="kestrel_sovereign.endpoints.models",
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "hOsTeD",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 500
        assert str(tmp_path) not in str(raised.value.detail)
        assert expected_log in caplog.text
        assert "hOsTeD" not in caplog.text
        assert MultiAgentConfig.from_file(config_path).agents["Hosted"] == replacement

    @pytest.mark.asyncio
    async def test_delete_endpoint_restores_registration_when_removal_is_refused(
        self,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        original = LocalAgentConfig(
            data_dir=Path("agent_data/hosted"),
            port=8801,
            autostart=True,
        )
        config = MultiAgentConfig(agents={"Hosted": original})
        config.save(config_path)
        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(return_value="did:test:hosted"),
            remove_agent=AsyncMock(return_value=False),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 404
        restored = MultiAgentConfig.from_file(config_path)
        assert restored.agents["Hosted"] == original

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        (
            RuntimeError("manager failed before shutdown"),
            TimeoutError("manager timed out before shutdown"),
            asyncio.CancelledError(),
        ),
        ids=("runtime-error", "timeout", "cancelled"),
    )
    async def test_delete_endpoint_restores_registration_when_agent_remains_live(
        self,
        tmp_path,
        failure,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        original = LocalAgentConfig(
            data_dir=Path("agent_data/hosted"),
            port=8801,
            autostart=True,
        )
        config = MultiAgentConfig(agents={"Hosted": original})
        config.save(config_path)
        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(return_value="did:test:hosted"),
            remove_agent=AsyncMock(side_effect=failure),
            get_agent=MagicMock(return_value=object()),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        with pytest.raises(type(failure)):
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        restored = MultiAgentConfig.from_file(config_path)
        assert restored.agents["Hosted"] == original
        assert request.app.state.multi_agent_config.agents["Hosted"] == original

    @pytest.mark.asyncio
    async def test_delete_endpoint_active_grouped_cancellation_restores_live_registration(
        self,
        tmp_path,
    ):
        """Cancellation precedence cannot skip synchronous config compensation."""

        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        original = LocalAgentConfig(
            data_dir=Path("agent_data/hosted"),
            port=8801,
            autostart=True,
        )
        config = MultiAgentConfig(agents={"Hosted": original})
        config.save(config_path)
        grouped = BaseExceptionGroup(
            "manager cancellation and failure",
            [asyncio.CancelledError(), RuntimeError("shutdown not completed")],
        )
        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(return_value="did:test:hosted"),
            remove_agent=AsyncMock(side_effect=grouped),
            get_agent=MagicMock(return_value=object()),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        with patch(
            "kestrel_sovereign.endpoints.models.asyncio.current_task",
            return_value=SimpleNamespace(cancelling=lambda: 1),
        ), pytest.raises(BaseExceptionGroup) as raised:
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value is grouped
        restored = MultiAgentConfig.from_file(config_path)
        assert restored.agents["Hosted"] == original
        assert request.app.state.multi_agent_config.agents["Hosted"] == original

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "process_failure",
        (KeyboardInterrupt(), SystemExit(17)),
        ids=("keyboard-interrupt", "system-exit"),
    )
    async def test_delete_endpoint_preserves_grouped_process_control_when_restore_fails(
        self,
        tmp_path,
        process_failure,
    ):
        """Compensation failure cannot replace a process-control group."""

        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        original = LocalAgentConfig(
            data_dir=Path("agent_data/hosted"),
            port=8801,
            autostart=True,
        )
        config = MultiAgentConfig(agents={"Hosted": original})
        config.save(config_path)
        retained = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id="did:test:process-control",
            runtime_path=tmp_path / "private-runtime",
            cause=OSError(f"private cleanup failure at {tmp_path}"),
        )
        terminal = BaseExceptionGroup(
            "manager process-control outcome",
            [retained, process_failure],
        )

        async def remove_config_then_fail(_name, **_kwargs):
            config_path.unlink()
            raise terminal

        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(
                return_value="did:test:process-control"
            ),
            remove_agent=AsyncMock(side_effect=remove_config_then_fail),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        with pytest.raises(BaseExceptionGroup) as raised:
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.exceptions[0] is terminal
        assert isinstance(raised.value.exceptions[1], FileNotFoundError)
        leaves = _exception_leaves(raised.value)
        assert retained in leaves
        assert process_failure in leaves
        assert str(tmp_path) not in str(raised.value)
        assert not config_path.exists()

    @pytest.mark.asyncio
    async def test_delete_endpoint_preserves_ordinary_failure_when_restore_fails(
        self,
        tmp_path,
    ):
        """The generic failure tail aggregates rather than masks either error."""

        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        original = LocalAgentConfig(
            data_dir=Path("agent_data/hosted"),
            port=8801,
            autostart=True,
        )
        config = MultiAgentConfig(agents={"Hosted": original})
        config.save(config_path)
        failure = RuntimeError(f"private manager failure at {tmp_path}")

        async def remove_config_then_fail(_name, **_kwargs):
            config_path.unlink()
            raise failure

        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(
                return_value="did:test:ordinary-failure"
            ),
            remove_agent=AsyncMock(side_effect=remove_config_then_fail),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        with pytest.raises(BaseExceptionGroup) as raised:
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.exceptions[0] is failure
        assert isinstance(raised.value.exceptions[1], FileNotFoundError)
        assert str(tmp_path) not in str(raised.value)
        assert not config_path.exists()

    @pytest.mark.asyncio
    async def test_delete_endpoint_preserves_cancellation_when_restore_fails(
        self,
        tmp_path,
    ):
        """Bare request cancellation retains precedence and compensation detail."""

        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        original = LocalAgentConfig(
            data_dir=Path("agent_data/hosted"),
            port=8801,
            autostart=True,
        )
        config = MultiAgentConfig(agents={"Hosted": original})
        config.save(config_path)
        cancelled = asyncio.CancelledError()

        async def remove_config_then_cancel(_name, **_kwargs):
            config_path.unlink()
            raise cancelled

        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(
                return_value="did:test:cancelled-failure"
            ),
            remove_agent=AsyncMock(side_effect=remove_config_then_cancel),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        with pytest.raises(BaseExceptionGroup) as raised:
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.exceptions[0] is cancelled
        assert isinstance(raised.value.exceptions[1], FileNotFoundError)
        assert not config_path.exists()

    @pytest.mark.asyncio
    async def test_delete_endpoint_reports_unpublished_agent_with_retained_runtime(
        self, tmp_path
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        retained = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id="did:pkh:retained",
            runtime_path=tmp_path / "runtime" / "tenant",
            cause=IsolatedRuntimeNamespaceError("foreign owner"),
        )
        manager = SimpleNamespace(remove_agent=AsyncMock(side_effect=retained))
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(agent_manager=manager))
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(request, "Hosted")

        assert raised.value.status_code == 409
        assert raised.value.detail == retained.metadata
        assert raised.value.detail["agent_removed"] is True
        assert raised.value.detail["runtime_retained"] is True
        assert "runtime_path" not in raised.value.detail
        assert str(tmp_path) not in str(raised.value.detail)

    @pytest.mark.asyncio
    async def test_delete_endpoint_extracts_retained_runtime_from_nested_group(
        self, tmp_path
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        class _PrivateRefundFailure(RuntimeError):
            pass

        retained = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id="did:pkh:compound-retained",
            runtime_path=tmp_path / "runtime" / "tenant",
            cause=IsolatedRuntimeNamespaceError("foreign owner"),
        )
        refund_failure = _PrivateRefundFailure(
            f"refund failed with secret at {tmp_path / 'wallet.json'}"
        )
        terminal = BaseExceptionGroup(
            "removal outcomes",
            [ExceptionGroup("cleanup", [retained]), refund_failure],
        )
        manager = SimpleNamespace(remove_agent=AsyncMock(side_effect=terminal))
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(agent_manager=manager))
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(request, "Hosted")

        assert raised.value.status_code == 409
        detail = raised.value.detail
        assert detail["code"] == "runtime_offboarding_retained"
        assert detail["agent_removed"] is True
        assert detail["runtime_retained"] is True
        assert detail["runtime_cleanup_pending"] is False
        assert detail["runtime_cleanup_state"] == "retained"
        assert detail["retained_agents"] == ["Hosted"]
        assert detail["compound_outcome"] is True
        assert detail["retained_outcome_count"] == 1
        assert detail["additional_outcome_count"] == 1
        assert detail["additional_outcome_types"] == ["RuntimeError"]
        assert "runtime_path" not in detail
        assert "wallet.json" not in str(detail)
        assert str(tmp_path) not in str(detail)

    @pytest.mark.asyncio
    async def test_delete_endpoint_reports_cleanup_timeout_as_pending_not_permanent(
        self, tmp_path
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        pending = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id="did:pkh:pending-cleanup",
            runtime_path=tmp_path / "runtime" / "tenant",
            cause=TimeoutError("private timeout detail"),
            cleanup_pending=True,
        )
        manager = SimpleNamespace(remove_agent=AsyncMock(side_effect=pending))
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(agent_manager=manager))
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(request, "Hosted")

        assert raised.value.status_code == 409
        assert raised.value.detail["runtime_cleanup_pending"] is True
        assert raised.value.detail["runtime_cleanup_state"] == "pending"
        assert "timeout detail" not in str(raised.value.detail)
        assert str(tmp_path) not in str(raised.value.detail)

    def test_retained_metadata_hides_private_internal_exception_type(self, tmp_path):
        class _PrivateOwnerMarkerMissing(IsolatedRuntimeNamespaceError):
            pass

        retained = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id="did:pkh:private-cause",
            runtime_path=tmp_path / "runtime" / "tenant",
            cause=_PrivateOwnerMarkerMissing("missing"),
        )

        assert retained.metadata["cause_type"] == "IsolatedRuntimeNamespaceError"

    @pytest.mark.asyncio
    async def test_remove_agent_refuses_foreign_runtime_owner_without_deleting(
        self, tmp_path
    ):
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:requested-offboarding"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(scope, "did:pkh:different-owner")
        retained = scope.path / "credential"
        retained.write_text("must-remain")
        agent = _make_mock_agent(did)
        agent.did = did
        agent.isolated_runtime_scope = scope
        manager._agents["Hosted"] = agent
        manager._agent_names[did] = "Hosted"
        config = LocalAgentConfig(data_dir="managed", port=8801)
        manager._seed_scheduler_authority({did: ("Hosted", config)})

        with pytest.raises(RuntimeOffboardingRetainedError) as raised:
            await manager.remove_agent(
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.metadata["runtime_retained"] is True
        assert raised.value.metadata["runtime_cleanup_pending"] is False
        assert raised.value.metadata["runtime_cleanup_state"] == "retained"
        assert raised.value.runtime_path == scope.path
        assert manager.get_agent("Hosted") is None
        assert manager.get_agent_name(did) is None
        assert not manager.is_scheduler_agent_authorized(did)
        assert retained.read_text() == "must-remain"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scheduler_authorized", (False, True))
    async def test_case_variant_remove_shuts_down_exact_live_agent_before_cleanup(
        self,
        monkeypatch,
        scheduler_authorized,
        tmp_path,
    ):
        from kestrel_sovereign.features import isolated_runtime

        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:case-variant-offboarding"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(
            scope,
            did,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        credential = scope.path / "feature_venvs" / "feature-safe" / "credential"
        credential.write_text("case-sensitive-custody")
        agent = _make_mock_agent(did)
        agent.did = did
        agent.isolated_runtime_scope = scope
        manager._agents["Hosted"] = agent
        manager._agent_names[did] = "Hosted"
        if scheduler_authorized:
            manager._seed_scheduler_authority(
                {
                    did: (
                        "Hosted",
                        LocalAgentConfig(data_dir="managed", port=8801),
                    )
                }
            )
        real_remove = isolated_runtime.remove_agent_runtime_namespace
        cleanup_calls = 0

        def assert_shutdown_precedes_cleanup(candidate):
            nonlocal cleanup_calls
            cleanup_calls += 1
            assert candidate is agent
            assert agent.shutdown.await_count == 1
            return real_remove(candidate)

        monkeypatch.setattr(
            isolated_runtime,
            "remove_agent_runtime_namespace",
            assert_shutdown_precedes_cleanup,
        )

        assert await manager.remove_agent(
            "hosted",
            offboard_runtime=True,
        )

        agent.shutdown.assert_awaited_once_with()
        assert cleanup_calls == 1
        assert manager.get_agent("Hosted") is None
        assert manager.get_agent("hosted") is None
        assert manager.get_agent_name(did) is None
        assert "Hosted" not in manager._agents
        assert not scope.path.exists()
        assert not manager.is_scheduler_agent_authorized(did)

    @pytest.mark.asyncio
    async def test_delete_case_edited_registration_cannot_bypass_live_shutdown(
        self,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:case-edited-registration"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(scope, did)
        agent = _make_mock_agent(did)
        agent.did = did
        agent.isolated_runtime_scope = scope
        manager._agents["Hosted"] = agent
        manager._agent_names[did] = "Hosted"
        config_path = tmp_path / "multi_agent.toml"
        config = MultiAgentConfig(
            agents={
                "hosted": LocalAgentConfig(
                    data_dir="agent_data/hosted",
                    port=8801,
                )
            }
        )
        config.save(config_path)
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        response = await delete_agent.__wrapped__(
            request,
            "HOSTED",
            offboard_runtime=True,
        )

        assert response["success"] is True
        agent.shutdown.assert_awaited_once_with()
        assert manager.get_agent("Hosted") is None
        assert manager.get_agent("hosted") is None
        assert not scope.path.exists()
        assert MultiAgentConfig.from_file(config_path).agents == {}
        assert request.app.state.multi_agent_config.agents == {}

    @pytest.mark.asyncio
    async def test_explicit_offboard_removes_authorized_cold_agent_runtime(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://host/kestrel")
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:cold-offboarding"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(
            scope,
            did,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        (scope.path / "feature_venvs" / "feature-safe" / "credential").write_text(
            "cold-credential"
        )
        manager._seed_scheduler_authority(
            {
                did: (
                    "Cold",
                    LocalAgentConfig(
                        data_dir="agent_data/cold",
                        port=8801,
                        autostart=False,
                    ),
                )
            }
        )

        assert await manager.remove_agent("Cold", offboard_runtime=True) is True

        assert not scope.path.exists()
        assert not manager.is_scheduler_agent_authorized(did)

    @pytest.mark.asyncio
    async def test_explicit_offboard_removes_witnessed_registered_cold_runtime(
        self,
        monkeypatch,
        tmp_path,
    ):
        """A known stopped registration is not a false 404 or duplicate sweep."""

        from kestrel_sovereign.features import isolated_runtime

        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://host/kestrel")
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:registered-cold-offboarding"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(
            scope,
            did,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        credential = scope.path / "feature_venvs" / "feature-safe" / "credential"
        credential.write_text("registered-cold-credential")
        real_remove = isolated_runtime.remove_isolated_runtime_namespace
        cleanup_calls = 0

        def count_cleanup(*args, **kwargs):
            nonlocal cleanup_calls
            cleanup_calls += 1
            return real_remove(*args, **kwargs)

        monkeypatch.setattr(
            isolated_runtime,
            "remove_isolated_runtime_namespace",
            count_cleanup,
        )

        config = LocalAgentConfig(data_dir="agent_data/cold", port=8801)
        legacy_root = config.resolve_data_dir(tmp_path) / "feature_venvs"
        legacy_feature = legacy_root / "NeverLoadedWhatsAppFeature"
        legacy_feature.mkdir(parents=True)
        legacy_root.chmod(0o700)
        legacy_feature.chmod(0o700)
        legacy_credential = legacy_feature / "auth" / "credentials.json"
        legacy_credential.parent.mkdir()
        legacy_credential.write_text("cold-legacy-credential")
        assert await manager.remove_agent(
            "Cold",
            offboard_runtime=True,
            known_agent_id=did,
            known_agent_config=config,
        )

        assert cleanup_calls == 1
        assert not scope.path.exists()
        assert not legacy_root.exists()
        assert manager.get_agent("Cold") is None
        assert not manager.is_scheduler_agent_authorized(did)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("hosted", "expected_state"),
        ((False, "not_hosted"), (True, "already_absent")),
        ids=("cold-storage-backed", "cold-hosted-already-absent"),
    )
    async def test_cold_identity_offboarding_preserves_typed_custody_outcome(
        self,
        monkeypatch,
        tmp_path,
        hosted,
        expected_state,
    ):
        """Cold cleanup reports the factory contract, never a synthesized path."""

        if hosted:
            monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
            monkeypatch.setenv(
                "KESTREL_DATABASE_URL", "postgresql://host/kestrel"
            )
        else:
            monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
            monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
        manager = AgentManager(base_data_dir=tmp_path)
        did = f"did:pkh:cold-custody-{expected_state}"
        config = LocalAgentConfig(data_dir="agent_data/cold", port=8801)
        config.resolve_data_dir(tmp_path).mkdir(parents=True)
        # A stale tree at the DID-derived location is not authority to delete it
        # when this manager's factory is storage-backed.
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        if not hosted:
            prepare_isolated_runtime_namespace(scope, did)
            retained = scope.path / "credential"
            retained.write_text("storage-backed-custody")

        with pytest.raises(RuntimeOffboardingNotPerformedError) as raised:
            await manager.remove_agent(
                "Cold",
                offboard_runtime=True,
                known_agent_id=did,
                known_agent_config=config,
            )

        assert raised.value.metadata["runtime_cleanup_state"] == expected_state
        assert raised.value.metadata["hosted_runtime_configured"] is hosted
        assert raised.value.metadata["runtime_already_absent"] is hosted
        if not hosted:
            assert retained.read_text() == "storage-backed-custody"

    @pytest.mark.asyncio
    async def test_delete_endpoint_cold_storage_agent_reports_not_hosted_custody(
        self,
        monkeypatch,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
        monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:cold-storage-endpoint"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(scope, did)
        retained = scope.path / "credential"
        retained.write_text("must-not-be-inferred-from-path")
        local = LocalAgentConfig(
            data_dir=Path("agent_data/cold-storage"),
            port=8801,
            autostart=False,
        )
        config = MultiAgentConfig(agents={"Cold": local})
        config_path = tmp_path / "multi_agent.toml"
        config.save(config_path)
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            AsyncMock(return_value=did),
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "Cold",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 409
        assert raised.value.detail["runtime_cleanup_state"] == "not_hosted"
        assert raised.value.detail["hosted_runtime_configured"] is False
        assert raised.value.detail["runtime_offboarded"] is False
        assert raised.value.detail["persisted_registration_removed"] is True
        assert retained.read_text() == "must-not-be-inferred-from-path"
        assert MultiAgentConfig.from_file(config_path).agents == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cleanup_outcome", ("success", "failure", "pending"))
    async def test_delete_endpoint_offboards_registered_unloaded_agent_truthfully(
        self,
        cleanup_outcome,
        monkeypatch,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent
        from kestrel_sovereign.features import isolated_runtime

        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://host/kestrel")
        manager = AgentManager(base_data_dir=tmp_path)
        did = f"did:pkh:cold-endpoint-{cleanup_outcome}"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(
            scope,
            did,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        credential = scope.path / "feature_venvs" / "feature-safe" / "credential"
        credential.write_text(f"cold-{cleanup_outcome}-credential")
        config_path = tmp_path / f"multi-agent-{cleanup_outcome}.toml"
        local = LocalAgentConfig(
            data_dir=Path("agent_data/cold"),
            port=8801,
            autostart=False,
        )
        config = MultiAgentConfig(agents={"Cold": local})
        config.save(config_path)
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )
        anchor = AsyncMock(return_value=did)
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            anchor,
        )
        real_remove = isolated_runtime.remove_isolated_runtime_namespace
        cleanup_calls = 0
        cleanup_started = threading.Event()
        cleanup_release = threading.Event()

        def controlled_cleanup(*args, **kwargs):
            nonlocal cleanup_calls
            cleanup_calls += 1
            if cleanup_outcome == "failure":
                raise PermissionError("private operator runtime path")
            if cleanup_outcome == "pending":
                cleanup_started.set()
                cleanup_release.wait(timeout=5)
            return real_remove(*args, **kwargs)

        monkeypatch.setattr(
            isolated_runtime,
            "remove_isolated_runtime_namespace",
            controlled_cleanup,
        )
        if cleanup_outcome == "pending":
            monkeypatch.setattr(
                "kestrel_sovereign.multi_agent.agent_manager.RUNTIME_OFFBOARD_TIMEOUT_S",
                0.01,
            )

        try:
            if cleanup_outcome == "success":
                result = await delete_agent.__wrapped__(
                    request,
                    "Cold",
                    offboard_runtime=True,
                )
                assert result["runtime_offboarded"] is True
                assert result["persisted_registration_removed"] is True
                assert not scope.path.exists()
            else:
                with pytest.raises(HTTPException) as raised:
                    await delete_agent.__wrapped__(
                        request,
                        "Cold",
                        offboard_runtime=True,
                    )
                assert raised.value.status_code == 409
                expected_state = (
                    "pending" if cleanup_outcome == "pending" else "retained"
                )
                assert raised.value.detail["runtime_cleanup_state"] == expected_state
                assert str(tmp_path) not in str(raised.value.detail)
                assert credential.read_text() == f"cold-{cleanup_outcome}-credential"

            assert cleanup_calls == 1
            assert MultiAgentConfig.from_file(config_path).agents == {}
            assert request.app.state.multi_agent_config.agents == {}
            assert manager.get_agent("Cold") is None
            anchor.assert_awaited_once_with(
                str(local.resolve_data_dir(tmp_path)),
                mode=AgentDIDLookupMode.COLD_READ_ONLY,
            )
        finally:
            cleanup_release.set()
            if cleanup_outcome == "pending":
                await manager.drain_quarantined_shutdowns()

    @pytest.mark.asyncio
    async def test_delete_endpoint_refuses_unresolved_registered_identity_before_mutation(
        self,
        monkeypatch,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        manager = AgentManager(base_data_dir=tmp_path)
        config_path = tmp_path / "multi_agent.toml"
        local = LocalAgentConfig(
            data_dir=Path("agent_data/unresolved"),
            port=8801,
            autostart=False,
        )
        config = MultiAgentConfig(agents={"Unresolved": local})
        config.save(config_path)
        before = config_path.read_bytes()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            AsyncMock(side_effect=OSError("private anchor path")),
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "Unresolved",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 409
        assert "identity is unavailable" in raised.value.detail
        assert config_path.read_bytes() == before
        assert "Unresolved" in request.app.state.multi_agent_config.agents
        assert manager._inflight_runtime_offboardings == {}

    @pytest.mark.asyncio
    async def test_offboarding_failure_and_cancellation_never_republish_dead_agent(
        self,
    ):
        manager = AgentManager()
        agent = _make_mock_agent("did:pkh:cancelled-offboarding")
        manager._agents["Hosted"] = agent
        manager._agent_names[agent.agent_id] = "Hosted"
        config = LocalAgentConfig(data_dir="managed", port=8801)
        manager._seed_scheduler_authority({agent.agent_id: ("Hosted", config)})
        retained = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id=agent.agent_id,
            runtime_path=Path("operator/runtime/tenant"),
            cause=IsolatedRuntimeNamespaceError("foreign owner"),
        )
        manager._finish_agent_runtime_offboarding = AsyncMock(
            return_value=(True, retained)
        )

        with pytest.raises(BaseExceptionGroup) as raised:
            await manager.remove_agent("Hosted", offboard_runtime=True)

        assert any(
            isinstance(error, asyncio.CancelledError)
            for error in raised.value.exceptions
        )
        assert any(
            isinstance(error, RuntimeOffboardingRetainedError)
            for error in raised.value.exceptions
        )
        assert manager.get_agent("Hosted") is None
        assert manager.get_agent_name(agent.agent_id) is None
        assert not manager.is_scheduler_agent_authorized(agent.agent_id)
        agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_offboarding_and_budget_failures_are_both_reported_after_unpublish(
        self,
    ):
        manager = AgentManager()
        agent = _make_mock_agent("did:pkh:offboard-and-refund")
        manager._agents["Hosted"] = agent
        manager._agent_names[agent.agent_id] = "Hosted"
        budget_entry = (object(), object())
        manager._child_budgets["Hosted"] = budget_entry
        retained = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id=agent.agent_id,
            runtime_path=Path("operator/runtime/tenant"),
            cause=IsolatedRuntimeNamespaceError("foreign owner"),
        )
        manager._finish_agent_runtime_offboarding = AsyncMock(
            return_value=(False, retained)
        )

        async def fail_refund(name: str) -> bool:
            assert name == "Hosted"
            assert manager._child_budgets[name] is budget_entry
            raise RuntimeError("refund failed")

        manager._release_child_budget_cancellation_safe = fail_refund

        with pytest.raises(ExceptionGroup) as raised:
            await manager.remove_agent("Hosted", offboard_runtime=True)

        assert any(
            isinstance(error, RuntimeOffboardingRetainedError)
            for error in raised.value.exceptions
        )
        assert any("refund failed" in str(error) for error in raised.value.exceptions)
        assert manager.get_agent("Hosted") is None
        assert manager.get_agent_name(agent.agent_id) is None
        assert manager._child_budgets["Hosted"] is budget_entry

    @pytest.mark.asyncio
    async def test_terminate_child_prunes_tracking_when_runtime_is_retained(self):
        manager = AgentManager()
        parent_did = "did:pkh:retained-parent"
        child = _make_mock_agent("did:pkh:retained-child")
        manager._agents["Child"] = child
        manager._agent_names[child.agent_id] = "Child"
        manager._parent_children[parent_did] = ["Child"]
        mandate = SpawnMandate(parent_did=parent_did, purpose="offboard")
        manager._child_mandates["Child"] = mandate
        retained = RuntimeOffboardingRetainedError(
            agent_name="Child",
            agent_id=child.agent_id,
            runtime_path=Path("operator/runtime/tenant"),
            cause=IsolatedRuntimeNamespaceError("foreign owner"),
        )
        manager._finish_agent_runtime_offboarding = AsyncMock(
            return_value=(False, retained)
        )

        with pytest.raises(RuntimeOffboardingRetainedError):
            await manager.terminate_child(
                parent_did,
                "Child",
                offboard_runtime=True,
            )

        assert manager.get_agent("Child") is None
        assert manager.get_children(parent_did) == []
        assert manager.get_mandate("Child") is None

    @pytest.mark.asyncio
    async def test_terminate_children_continues_after_retained_and_grouped_outcomes(
        self,
    ):
        manager = AgentManager()
        parent_did = "did:pkh:cascade-parent"
        manager._parent_children[parent_did] = ["First", "Second", "Third"]
        for name in ("First", "Second", "Third"):
            manager._child_mandates[name] = SpawnMandate(
                parent_did=parent_did, purpose="cascade"
            )
        retained = RuntimeOffboardingRetainedError(
            agent_name="First",
            agent_id="did:pkh:first",
            runtime_path=Path("operator/runtime/first"),
            cause=OSError("retained"),
        )
        grouped = BaseExceptionGroup(
            "cancelled retained cleanup",
            [
                asyncio.CancelledError(),
                RuntimeOffboardingRetainedError(
                    agent_name="Second",
                    agent_id="did:pkh:second",
                    runtime_path=Path("operator/runtime/second"),
                    cause=OSError("retained"),
                ),
            ],
        )
        manager.remove_agent = AsyncMock(side_effect=[retained, grouped, True])

        with pytest.raises(BaseExceptionGroup) as raised:
            await manager.terminate_children(
                parent_did,
                offboard_runtime=True,
            )

        assert manager.remove_agent.await_args_list == [
            (("First",), {"offboard_runtime": True}),
            (("Second",), {"offboard_runtime": True}),
            (("Third",), {"offboard_runtime": True}),
        ]
        assert _exception_group_contains(
            raised.value, RuntimeOffboardingRetainedError
        )
        assert _exception_group_contains(raised.value, asyncio.CancelledError)
        assert manager.get_children(parent_did) == []

    @pytest.mark.asyncio
    async def test_terminate_child_still_removes_parent_after_descendant_retention(
        self,
    ):
        manager = AgentManager()
        parent_did = "did:pkh:root-parent"
        child = _make_mock_agent("did:pkh:cascade-child")
        manager._agents["Child"] = child
        manager._agent_names[child.agent_id] = "Child"
        manager._parent_children[parent_did] = ["Child"]
        manager._child_mandates["Child"] = SpawnMandate(
            parent_did=parent_did, purpose="cascade"
        )
        descendant_retained = RuntimeOffboardingRetainedError(
            agent_name="Grandchild",
            agent_id="did:pkh:grandchild",
            runtime_path=Path("operator/runtime/grandchild"),
            cause=OSError("retained"),
        )
        manager.terminate_children = AsyncMock(side_effect=descendant_retained)

        async def remove_child(name: str, *, offboard_runtime: bool) -> bool:
            assert name == "Child"
            assert offboard_runtime is True
            manager._agents.pop(name)
            manager._agent_names.pop(child.agent_id)
            return True

        manager.remove_agent = AsyncMock(side_effect=remove_child)

        with pytest.raises(RuntimeOffboardingRetainedError):
            await manager.terminate_child(
                parent_did,
                "Child",
                offboard_runtime=True,
            )

        manager.remove_agent.assert_awaited_once_with(
            "Child", offboard_runtime=True
        )
        assert manager.get_children(parent_did) == []

    @pytest.mark.asyncio
    async def test_terminate_child_types_post_removal_reconciliation_failure(self):
        manager = AgentManager()
        parent_did = "did:pkh:reconcile-parent"
        manager._parent_children[parent_did] = ["Child"]
        manager.remove_agent = AsyncMock(return_value=True)
        cause = OSError("private reconciliation path /operator/runtime")
        manager._prune_child_tracking_if_fully_removed = AsyncMock(
            side_effect=cause
        )

        with pytest.raises(ChildTerminationReconciliationError) as raised:
            await manager.terminate_child(
                parent_did,
                "Child",
                offboard_runtime=True,
            )

        assert raised.value.cause is cause
        assert raised.value.metadata == {
            "code": "child_termination_reconciliation_failed",
            "child_name": "Child",
            "cause_type": "OSError",
        }
        assert "/operator/runtime" not in str(raised.value)
        manager.remove_agent.assert_awaited_once_with(
            "Child", offboard_runtime=True
        )

    @pytest.mark.asyncio
    async def test_remove_agent_revokes_live_scheduler_authority_before_future_cold_wake(
        self,
    ):
        """DELETE cannot be undone by the static startup config's old DID map."""
        manager = AgentManager()
        config = LocalAgentConfig(data_dir="managed", port=8801)
        mock = _make_mock_agent("did:pkh:removed")
        manager._agents["Managed"] = mock
        manager._agent_names[mock.agent_id] = "Managed"
        manager._seed_scheduler_authority({mock.agent_id: ("Managed", config)})

        assert await manager.remove_agent("Managed") is True
        assert not manager.is_scheduler_agent_authorized(mock.agent_id)
        manager._initialize_agent = AsyncMock()
        with pytest.raises(LookupError, match="no longer authorized"):
            await manager.load_agent(
                "Managed",
                config,
                expected_agent_id=mock.agent_id,
            )
        manager._initialize_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_unloaded_scheduler_agent_revokes_before_executor_wake(self):
        """DELETE of a configured cold tenant is a completed runtime removal."""
        manager = AgentManager()
        agent_id = "did:pkh:unloaded-delete"
        config = LocalAgentConfig(data_dir="cold", port=8801, autostart=False)
        manager._seed_scheduler_authority({agent_id: ("Cold", config)})
        manager._initialize_agent = AsyncMock()
        executor = AgentManagerHostedSchedulerExecutor(
            manager,
            {agent_id: ("Cold", config)},
        )
        execution = SchedulerExecution(
            id="execution-unloaded-delete",
            schedule_id="schedule-unloaded-delete",
            agent_id=agent_id,
            task_name="test_task",
            args={},
            scheduled_for="2026-07-25T00:00:00+00:00",
            idempotency_key="effect-unloaded-delete",
            attempt=1,
            owner="host",
        )

        assert await manager.remove_agent("Cold") is True
        assert not manager.is_scheduler_agent_authorized(agent_id)
        with pytest.raises(LookupError, match="No hosted agent configuration"):
            await executor.execute_scheduled(execution)
        manager._initialize_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_remove_restores_scheduler_authority(self):
        """A failed DELETE must not silently change desired state."""
        manager = AgentManager()
        config = LocalAgentConfig(data_dir="managed", port=8801)
        mock = _make_mock_agent("did:pkh:still-live")
        mock.shutdown.side_effect = RuntimeError("shutdown failed")
        manager._agents["Managed"] = mock
        manager._agent_names[mock.agent_id] = "Managed"
        manager._seed_scheduler_authority({mock.agent_id: ("Managed", config)})

        assert await manager.remove_agent("Managed") is False
        assert manager.is_scheduler_agent_authorized(mock.agent_id)

    @pytest.mark.asyncio
    async def test_delete_serializes_with_inflight_scheduler_lifecycle_lock(self):
        """DELETE waits for an in-flight dispatch lease, then revokes cold wake."""
        manager = AgentManager()
        config = LocalAgentConfig(data_dir="managed", port=8801)
        mock = _make_mock_agent("did:pkh:locked-delete")
        manager._agents["Managed"] = mock
        manager._agent_names[mock.agent_id] = "Managed"
        manager._seed_scheduler_authority({mock.agent_id: ("Managed", config)})
        dispatch_started = asyncio.Event()
        allow_dispatch_finish = asyncio.Event()

        async def in_flight_dispatch():
            async with manager.scheduler_lifecycle_lock(mock.agent_id):
                dispatch_started.set()
                await allow_dispatch_finish.wait()

        dispatch = asyncio.create_task(in_flight_dispatch())
        await asyncio.wait_for(dispatch_started.wait(), timeout=1)
        deletion = asyncio.create_task(manager.remove_agent("Managed"))
        await asyncio.sleep(0)
        assert not deletion.done()
        assert manager.is_scheduler_agent_authorized(mock.agent_id)

        allow_dispatch_finish.set()
        await dispatch
        assert await deletion is True
        assert not manager.is_scheduler_agent_authorized(mock.agent_id)

    @pytest.mark.asyncio
    async def test_stale_delete_never_removes_same_name_replacement(self):
        """A DELETE holding DID A's lock cannot unpublish replacement DID B."""

        manager = AgentManager()
        old_config = LocalAgentConfig(data_dir="old", port=8801)
        new_config = LocalAgentConfig(data_dir="new", port=8802)
        old = _make_mock_agent("did:pkh:old-delete-target")
        replacement = _make_mock_agent("did:pkh:same-name-replacement")
        manager._agents["Managed"] = old
        manager._agent_names[old.agent_id] = "Managed"
        manager._seed_scheduler_authority(
            {old.agent_id: ("Managed", old_config)}
        )

        lifecycle_lock = manager.scheduler_lifecycle_lock(old.agent_id)
        await lifecycle_lock.acquire_read()
        deletion = asyncio.create_task(manager.remove_agent("Managed"))
        try:
            while lifecycle_lock._waiting_writers == 0:
                await asyncio.sleep(0)

            async with manager._a2a_lifecycle_lock:
                manager._agents["Managed"] = replacement
                manager._agent_names.pop(old.agent_id)
                manager._agent_names[replacement.agent_id] = "Managed"
                manager._revoke_scheduler_authority("Managed", old.agent_id)
                manager._scheduler_revoked_names.discard("Managed")
                manager._scheduler_authority_by_did[replacement.agent_id] = (
                    "Managed",
                    new_config,
                )
                manager._scheduler_authority_by_name["Managed"] = replacement.agent_id
                manager._scheduler_execution_scope.add(replacement.agent_id)
        finally:
            lifecycle_lock.release_read()

        assert await asyncio.wait_for(deletion, timeout=1) is False
        assert manager.get_agent("Managed") is replacement
        replacement.shutdown.assert_not_awaited()
        assert manager.is_scheduler_agent_authorized(replacement.agent_id)

    @pytest.mark.asyncio
    async def test_hosted_effects_share_lifecycle_read_lease_before_delete(self):
        """Sibling schedules overlap, while DELETE drains both before revoking."""

        manager = AgentManager()
        config = LocalAgentConfig(data_dir="managed", port=8801)
        mock = _make_mock_agent("did:pkh:shared-scheduler-effects")
        manager._agents["Managed"] = mock
        manager._agent_names[mock.agent_id] = "Managed"
        manager._seed_scheduler_authority({mock.agent_id: ("Managed", config)})
        effects_started: list[str] = []
        both_started = asyncio.Event()
        release_effects = asyncio.Event()

        async def dispatch(_task_name, args):
            effects_started.append(args["effect"])
            if len(effects_started) == 2:
                both_started.set()
            await release_effects.wait()
            return "dispatched"

        mock.features = {
            "SchedulerFeature": SimpleNamespace(
                _dispatch_scheduled_task=dispatch,
            )
        }
        executor = AgentManagerHostedSchedulerExecutor(manager)

        def execution(effect: str) -> SchedulerExecution:
            return SchedulerExecution(
                id=f"execution-{effect}",
                schedule_id=f"schedule-{effect}",
                agent_id=mock.agent_id,
                task_name="test_task",
                args={"effect": effect},
                scheduled_for="2026-07-25T00:00:00+00:00",
                idempotency_key=f"effect-{effect}",
                attempt=1,
                owner="host",
            )

        scheduled = [
            asyncio.create_task(executor.execute_scheduled(execution(effect)))
            for effect in ("a", "b")
        ]
        deletion = None
        try:
            await asyncio.wait_for(both_started.wait(), timeout=1)
            assert set(effects_started) == {"a", "b"}

            deletion = asyncio.create_task(manager.remove_agent("Managed"))
            await asyncio.sleep(0.02)
            assert not deletion.done()
            assert manager.is_scheduler_agent_authorized(mock.agent_id)

            release_effects.set()
            assert await asyncio.gather(*scheduled) == ["dispatched", "dispatched"]
            assert await asyncio.wait_for(deletion, timeout=1) is True
            assert not manager.is_scheduler_agent_authorized(mock.agent_id)
        finally:
            release_effects.set()
            for task in (*scheduled, deletion):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *scheduled,
                *(() if deletion is None else (deletion,)),
                return_exceptions=True,
            )

    @pytest.mark.asyncio
    async def test_hosted_executor_uses_live_same_did_replacement_after_handoff(
        self,
    ):
        """A writer replacement cannot leave a stale warm agent dispatchable."""

        manager = AgentManager()
        config = LocalAgentConfig(data_dir="managed", port=8801)
        agent_id = "did:pkh:scheduler-replacement"
        original = _make_mock_agent(agent_id)
        replacement = _make_mock_agent(agent_id)
        original_dispatch = AsyncMock(return_value="original")
        replacement_dispatch = AsyncMock(return_value="replacement")
        original.features = {
            "SchedulerFeature": SimpleNamespace(
                _dispatch_scheduled_task=original_dispatch,
            )
        }
        replacement.features = {
            "SchedulerFeature": SimpleNamespace(
                _dispatch_scheduled_task=replacement_dispatch,
            )
        }
        manager._agents["Managed"] = original
        manager._agent_names[agent_id] = "Managed"
        manager._seed_scheduler_authority({agent_id: ("Managed", config)})
        warm_lookup_complete = asyncio.Event()
        allow_reader_admission = asyncio.Event()

        class HandoffProbeExecutor(AgentManagerHostedSchedulerExecutor):
            def _execution_lease_for(self, target_agent_id):
                base_lease = super()._execution_lease_for(target_agent_id)

                @asynccontextmanager
                async def delayed_reader_lease():
                    warm_lookup_complete.set()
                    await allow_reader_admission.wait()
                    async with base_lease:
                        yield

                return delayed_reader_lease()

        executor = HandoffProbeExecutor(manager)
        execution = SchedulerExecution(
            id="execution-replacement",
            schedule_id="schedule-replacement",
            agent_id=agent_id,
            task_name="test_task",
            args={},
            scheduled_for="2026-07-25T00:00:00+00:00",
            idempotency_key="effect-replacement",
            attempt=1,
            owner="host",
        )
        scheduled = asyncio.create_task(executor.execute_scheduled(execution))
        try:
            await asyncio.wait_for(warm_lookup_complete.wait(), timeout=1)
            async with manager.scheduler_lifecycle_lock(agent_id):
                manager._agents["Managed"] = replacement
            allow_reader_admission.set()

            assert await asyncio.wait_for(scheduled, timeout=1) == "replacement"
            original_dispatch.assert_not_awaited()
            replacement_dispatch.assert_awaited_once_with("test_task", {})
        finally:
            allow_reader_admission.set()
            if not scheduled.done():
                scheduled.cancel()
            await asyncio.gather(scheduled, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_delete_serializes_real_executor_cold_wake_before_registration(
        self,
    ):
        """DELETE cannot return 404 and lose a cold wake already holding its DID lock."""
        manager = AgentManager()
        agent_id = "did:pkh:cold-delete-race"
        config = LocalAgentConfig(data_dir="cold", port=8801, autostart=False)
        manager._seed_scheduler_authority({agent_id: ("Cold", config)})

        initialization_started = asyncio.Event()
        allow_initialization = asyncio.Event()
        dispatch_started = asyncio.Event()
        allow_dispatch_finish = asyncio.Event()

        async def dispatch(_task_name, _args):
            dispatch_started.set()
            await allow_dispatch_finish.wait()
            return "dispatched"

        cold = _make_mock_agent(agent_id)
        cold.features = {
            "SchedulerFeature": SimpleNamespace(
                _dispatch_scheduled_task=dispatch,
            )
        }

        async def initialize(_name, _config, **kwargs):
            assert kwargs == {"scheduler_lifecycle_lock_held": True}
            initialization_started.set()
            await allow_initialization.wait()
            return cold

        manager._initialize_agent = AsyncMock(side_effect=initialize)
        executor = AgentManagerHostedSchedulerExecutor(
            manager,
            {agent_id: ("Cold", config)},
        )
        execution = SchedulerExecution(
            id="execution-cold-delete-race",
            schedule_id="schedule-cold-delete-race",
            agent_id=agent_id,
            task_name="test_task",
            args={},
            scheduled_for="2026-07-25T00:00:00+00:00",
            idempotency_key="effect-cold-delete-race",
            attempt=1,
            owner="host",
        )

        scheduled = asyncio.create_task(executor.execute_scheduled(execution))
        await asyncio.wait_for(initialization_started.wait(), timeout=1)

        deletion = asyncio.create_task(manager.remove_agent("Cold"))
        await asyncio.sleep(0)
        assert not deletion.done()
        assert manager.is_scheduler_agent_authorized(agent_id)

        allow_initialization.set()
        await asyncio.wait_for(dispatch_started.wait(), timeout=1)
        assert manager.get_agent("Cold") is cold
        assert not deletion.done()

        allow_dispatch_finish.set()
        assert await asyncio.wait_for(scheduled, timeout=1) == "dispatched"
        assert await asyncio.wait_for(deletion, timeout=1) is True
        assert manager.get_agent("Cold") is None
        assert not manager.is_scheduler_agent_authorized(agent_id)
        cold.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shared_pg_hosted_cold_wake_cancellation_releases_delete_waiter(
        self,
        monkeypatch,
        tmp_path,
    ):
        """A configured cold wake skips dynamic reacquire and cancellation drains."""

        manager = AgentManager(base_data_dir=tmp_path)
        manager.set_scheduler_polling_managed_by_host(True)
        agent_id = "did:scheduler:shared-pg-cold-cancel"
        config = LocalAgentConfig(
            data_dir="cold",
            port=8801,
            autostart=False,
        )
        manager._seed_scheduler_authority({agent_id: ("Cold", config)})
        initialization_started = asyncio.Event()
        shutdown_finished = asyncio.Event()
        registration_hook = AsyncMock(
            side_effect=AssertionError(
                "configured cold wake must not enter dynamic registration"
            )
        )
        manager.set_scheduler_tenant_registration_hook(registration_hook)

        class ColdAgent:
            def __init__(self, *, did, **_kwargs):
                self.did = did
                self.agent_id = did
                self.features = {}

            async def initialize(self):
                initialization_started.set()
                await asyncio.Event().wait()

            async def shutdown(self):
                shutdown_finished.set()

        class TestLLMService:
            # Mirrors the real constructor envelope: each agent's service is
            # bound to that agent's own data root, so per-agent usage rows
            # cannot collapse into one shared database (#2769).
            def __init__(self, database_url=None, agent_data_dir=None):
                self.agent_data_dir = agent_data_dir

            async def close(self):
                return None

        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv(
            "KESTREL_DATABASE_URL",
            "postgresql://scheduler-cold-test",
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            AsyncMock(return_value=agent_id),
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.KestrelAgent",
            ColdAgent,
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.LLMService",
            TestLLMService,
        )
        monkeypatch.setattr(
            LocalAgentConfig,
            "validate_runtime",
            lambda self, **_kwargs: [],
        )
        executor = AgentManagerHostedSchedulerExecutor(manager)
        execution = SchedulerExecution(
            id="execution-shared-pg-cold-cancel",
            schedule_id="schedule-shared-pg-cold-cancel",
            agent_id=agent_id,
            task_name="wait_reconcile",
            args={},
            scheduled_for="2026-07-25T00:00:00+00:00",
            idempotency_key="effect-shared-pg-cold-cancel",
            attempt=1,
            owner="host",
        )

        scheduled = asyncio.create_task(executor.execute_scheduled(execution))
        deletion = None
        try:
            await asyncio.wait_for(initialization_started.wait(), timeout=1)
            assert manager.scheduler_lifecycle_lock(agent_id).locked()

            deletion = asyncio.create_task(manager.remove_agent("Cold"))
            await asyncio.sleep(0)
            assert not deletion.done()
            assert manager.is_scheduler_agent_authorized(agent_id)

            scheduled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await scheduled
            assert shutdown_finished.is_set()
            assert await asyncio.wait_for(deletion, timeout=1) is True
            assert manager.scheduler_authority_for(agent_id) is None
            assert not manager.scheduler_lifecycle_lock(agent_id).locked()
            registration_hook.assert_not_awaited()
        finally:
            if not scheduled.done():
                scheduled.cancel()
            if deletion is not None and not deletion.done():
                deletion.cancel()
            await asyncio.gather(
                scheduled,
                *(
                    (deletion,)
                    if deletion is not None
                    else ()
                ),
                return_exceptions=True,
            )

    @pytest.mark.asyncio
    async def test_scheduler_cold_wake_receives_host_a2a_and_feature_route_onboarding(
        self,
        tmp_path,
    ):
        """A scheduler-loaded tenant is integrated like an autostart tenant."""
        import kestrel_sovereign.endpoints.agent as agent_endpoint
        from kestrel_sovereign import server
        from kestrel_sovereign.a2a.did_registry import install_a2a_did_resolver
        from kestrel_sovereign.a2a.envelope_signing import (
            bound_envelope_fields,
            canonical_message,
            sign_envelope,
            verify_inbound_envelope,
        )
        from kestrel_sovereign.a2a.inbound_authorization import (
            has_a2a_inbound_scoped_policy,
            install_a2a_inbound_sender_authorizer,
        )
        from kestrel_sovereign.a2a.types import Message, TaskSendParams, TextPart
        from kestrel_sovereign.features.peers.directory import PeerRequester
        from kestrel_sovereign.identity.did_web import build_verification_methods
        from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair

        host = FastAPI()
        manager = AgentManager()
        host.state.agent_manager = manager
        host.state.agent = None
        host.state.demo_mode = False

        warm_did = "did:web:example.test:agent:warm"
        warm_keypair = generate_hybrid_keypair()
        warm = SimpleNamespace(
            agent_id=warm_did,
            identity=SimpleNamespace(
                is_hybrid=True,
                signing_did=warm_did,
                new_verification_methods=build_verification_methods(
                    warm_did, warm_keypair.public_keys()
                ),
            ),
            features={},
            shutdown=AsyncMock(),
        )
        manager._register_agent("Warm", warm)
        # Model initial fleet onboarding: a later cold wake must not replace
        # this warm recipient's verification/authorization seams.
        install_a2a_did_resolver(manager, recipient=warm)
        install_a2a_inbound_sender_authorizer(manager, recipient=warm)
        warm_resolver = warm.a2a_did_resolver.__self__
        warm_authorizer = warm.a2a_inbound_sender_authorizer

        router = APIRouter()

        @router.get("/cold-only")
        async def cold_only():
            return {"cold": True}

        feature = SimpleNamespace(
            enabled=True,
            receiver=None,
            get_router=lambda: router,
        )
        cold_did = "did:web:example.test:agent:cold"
        cold_keypair = generate_hybrid_keypair()
        # Production-shaped cold KestrelAgent: the local Peers adapter is
        # feature-internal, so these injected hosted fields are genuinely None.
        cold = KestrelAgent(
            did=cold_did,
            storage_path=str(tmp_path / "cold" / "kestrel_prime.db"),
        )
        cold.identity = SimpleNamespace(
            is_hybrid=True,
            signing_did=cold_did,
            new_verification_methods=build_verification_methods(
                cold_did, cold_keypair.public_keys()
            ),
        )
        # Normal AgentManager construction leaves these public injection attrs
        # empty. Its PeersFeature owns the live local-host route instead.
        # Onboarding must bind the immutable manager policy to this pair.
        live_router = SimpleNamespace(
            authorize_inbound_sender=AsyncMock(return_value=True),
        )
        live_requester = PeerRequester(cold_did, object())
        live_peers_feature = SimpleNamespace(
            hosted_peer_directory_context=lambda: (live_router, live_requester),
            get_router=lambda: None,
        )
        cold.features = {
            "ColdOnly": feature,
            "PeersFeature": live_peers_feature,
        }
        cold.peer_directory_router = None
        cold.peer_requester = None
        cold.task_manager = SimpleNamespace(
            create_task=AsyncMock(return_value=SimpleNamespace(id="cold-wake-a2a")),
        )
        cold.shutdown = AsyncMock()
        cold.wait_for_shutdown_completion = None
        cold._set_display_name = lambda _name: None
        config = LocalAgentConfig(data_dir="cold", port=8802, autostart=False)
        manager._seed_scheduler_authority({cold.agent_id: ("Cold", config)})
        manager._initialize_agent = AsyncMock(return_value=cold)
        manager.set_agent_registration_hook(
            lambda name, agent: server._onboard_host_registered_agent(
                host, manager, name, agent
            )
        )

        loaded = await manager.load_agent(
            "Cold", config, expected_agent_id=cold.agent_id
        )

        assert loaded is cold
        assert cold.a2a_did_resolver(warm_did)["id"] == warm_did
        assert warm.a2a_did_resolver(cold_did)["id"] == cold_did
        assert warm.a2a_did_resolver.__self__ is warm_resolver
        assert cold.a2a_did_resolver.__self__ is not warm.a2a_did_resolver.__self__
        assert warm.a2a_inbound_sender_authorizer is warm_authorizer
        assert cold.a2a_inbound_sender_authorizer is not warm_authorizer
        assert has_a2a_inbound_scoped_policy(cold) is True
        assert cold.a2a_inbound_sender_authorizer.requires_verified_sender is True
        assert cold.a2a_inbound_sender_authorizer.has_valid_current_scope() is False
        assert cold._a2a_host_manager is manager
        hosted_policy = manager.a2a_hosted_policy_for(cold)
        assert hosted_policy is not None
        assert hosted_policy.router is live_router
        assert hosted_policy.requester is live_requester

        # Exercise the actual verification seam, not merely resolver lookup:
        # a signed warm→cold same-host envelope must validate after the cold
        # tenant is registered through the scheduler path.
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata = {"sender": warm_did}
        metadata["signature"] = sign_envelope(
            warm_keypair,
            sender=warm_did,
            task_id="cold-wake-a2a",
            message="scheduler woke cold peer",
            timestamp=timestamp,
            session_id="cold-wake-session",
            bound=bound_envelope_fields(metadata),
        )
        verdict = await verify_inbound_envelope(
            metadata,
            task_id="cold-wake-a2a",
            message="scheduler woke cold peer",
            session_id="cold-wake-session",
            resolver=cold.a2a_did_resolver,
            require_signed=True,
        )
        assert verdict.ok is True and verdict.verified is True

        # Exercise the recipient's real verified-send path under the manager
        # lease. The raw agent attrs remain None, so this would fail if the
        # hosted policy had not captured PeersFeature's effective context.
        send_metadata = {"sender": warm_did}
        send_metadata["signature"] = sign_envelope(
            warm_keypair,
            sender=warm_did,
            task_id="cold-wake-a2a",
            message=canonical_message(["scheduler woke cold peer"]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id="cold-wake-session",
            bound=bound_envelope_fields(send_metadata),
        )
        params = TaskSendParams(
            id="cold-wake-a2a",
            sessionId="cold-wake-session",
            message=Message(
                role="user", parts=[TextPart(text="scheduler woke cold peer")],
            ),
            metadata=send_metadata,
        )
        created = await agent_endpoint._create_verified_a2a_task(
            cold,
            params,
            params.message.parts,
            [],
            [],
        )
        assert created.id == "cold-wake-a2a"
        live_router.authorize_inbound_sender.assert_awaited_once_with(
            live_requester, warm_did,
        )
        assert cold.task_manager.create_task.await_count == 1
        with TestClient(host) as client:
            response = client.get("/cold-only")
            assert response.status_code == 200
            assert await manager.remove_agent("Cold") is True
            deleted_response = client.get("/cold-only")
        assert response.status_code == 200
        assert response.json() == {"cold": True}
        assert deleted_response.status_code == 404

    @pytest.mark.asyncio
    async def test_remove_nonexistent_agent(self):
        manager = AgentManager()
        result = await manager.remove_agent("ghost")
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        manager = AgentManager()
        agent1 = _make_mock_agent("did:1")
        agent2 = _make_mock_agent("did:2")
        manager._agents["A"] = agent1
        manager._agents["B"] = agent2
        manager._agent_names["did:1"] = "A"
        manager._agent_names["did:2"] = "B"

        await manager.shutdown_all()
        assert len(manager._agents) == 0
        agent1.shutdown.assert_awaited_once()
        agent2.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remove_agent_quarantines_deferred_durable_close_before_unpublishing(
        self, tmp_path
    ):
        """One production removal call drains the real SQLite worker (#2713)."""
        from kestrel_sovereign.signals import (
            OrderedLockManager,
            SignalDispatcher,
            SignalLogStore,
            SourceRegistry,
        )
        from kestrel_sovereign.storage.db import SQLiteBackend

        manager = AgentManager()
        agent = KestrelAgent(
            did="did:test:manager-durable-shutdown",
            storage_path=str(tmp_path / "agent.db"),
        )
        backend = SQLiteBackend(str(tmp_path / "ledger.db"))
        await backend.connect()
        worker = aiosqlite_worker(backend._connection)
        log_store = SignalLogStore(backend)
        await log_store.initialize()
        dispatcher = SignalDispatcher(
            agent=agent,
            registry=SourceRegistry(),
            lock_manager=OrderedLockManager(),
            store=log_store,
        )
        await dispatcher.initialize_durable_delivery()
        agent.dispatcher = dispatcher

        original_release = dispatcher._durable_store.release_initial_reservations
        release_entered = asyncio.Event()
        allow_release = asyncio.Event()
        storage_closed: list[bool] = []

        async def block_release(*args, **kwargs):
            release_entered.set()
            await allow_release.wait()
            return await original_release(*args, **kwargs)

        class _Storage:
            async def close(self):
                storage_closed.append(True)
                await backend.close()

        dispatcher._durable_store.release_initial_reservations = block_release
        agent.features = {}
        agent.llm_service = None
        agent.task_manager = None
        agent.memory_system = None
        agent._sync_service = None
        agent.storage = _Storage()
        manager._agents["Managed"] = agent
        manager._agent_names[agent.agent_id] = "Managed"

        try:
            # The manager's normal outer timeout cancels the bounded agent
            # shutdown. It withdraws the routing entry within that advertised
            # bound, while a retained quarantine reaper remains the sole owner
            # of the dispatcher release and SQLite close.
            with patch(
                "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT",
                0.05,
            ):
                remove_task = asyncio.create_task(manager.remove_agent("Managed"))
                await asyncio.wait_for(release_entered.wait(), timeout=1.0)
                assert await asyncio.wait_for(remove_task, timeout=1.0) is True

                assert manager.get_agent("Managed") is None
                assert storage_closed == []
                quarantined = manager.quarantined_shutdowns()
                assert len(quarantined) == 1
                assert next(iter(quarantined.values()))["pending"] is True

                allow_release.set()
                # A reaper owns a real SQLite worker, whose completion is not
                # ordered by a count of event-loop turns. Join the manager's
                # explicit terminal lifecycle boundary instead of letting test
                # teardown close that worker's backend underneath its release.
                assert await asyncio.wait_for(
                    manager.drain_quarantined_shutdowns(), timeout=1.0
                ) is False

            assert storage_closed == [True]
            assert backend._connection is None
            for _ in range(100):
                if not worker.is_alive():
                    break
                await asyncio.sleep(0)
            assert not worker.is_alive()
        finally:
            allow_release.set()
            dispatcher._durable_store.release_initial_reservations = original_release
            # Keep the production-shaped owner alive even when an assertion
            # above fails.  Closing the backend first would manufacture the
            # very post-close owner-release race this regression covers.
            if manager._quarantined_shutdown_reapers:
                await asyncio.wait_for(
                    manager.drain_quarantined_shutdowns(), timeout=1.0
                )
            if backend._connection is not None:
                await backend.close()

    @pytest.mark.asyncio
    async def test_shutdown_handles_errors(self):
        """Shutdown should continue even if one agent errors."""
        manager = AgentManager()
        agent1 = _make_mock_agent("did:1")
        agent1.shutdown = AsyncMock(side_effect=Exception("boom"))
        agent2 = _make_mock_agent("did:2")
        manager._agents["A"] = agent1
        manager._agents["B"] = agent2
        manager._agent_names["did:1"] = "A"
        manager._agent_names["did:2"] = "B"

        with pytest.raises(ExceptionGroup, match="fleet agents failed"):
            await manager.shutdown_all()
        # Both are attempted, but a failed shutdown stays published. Removing
        # it would discard the lifecycle owner before durable cleanup can be
        # confirmed on a later retry. The aggregate is raised only after B has
        # received its own cleanup attempt.
        assert set(manager._agents) == {"A"}
        assert manager.get_agent("B") is None
        agent1.shutdown.assert_awaited_once()
        agent2.shutdown.assert_awaited_once()


class TestLoadFromConfig:
    """Test loading agents from MultiAgentConfig."""

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_future_agent_receives_bound_host_context_registry(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
    ):
        mock_get_did.return_value = "did:future"
        future_agent = _make_mock_agent("did:future")
        mock_agent_cls.return_value = future_agent
        manager = AgentManager(base_data_dir=tmp_path)
        registry = object()
        publication_gate = asyncio.Event()
        manager.bind_host_context_clause_registry(registry)
        manager.set_host_context_publication_gate(publication_gate)

        async def initialize_with_gate_bound():
            assert future_agent._host_context_publication_gate is publication_gate
            assert (
                future_agent._host_context_publication_state
                is manager._host_context_publication_state
            )
            assert future_agent._host_context_publication_generation is None
            assert not publication_gate.is_set()

        future_agent.initialize.side_effect = initialize_with_gate_bound

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent(
                "future",
                LocalAgentConfig(data_dir=Path("future"), port=8801),
            )

        assert (
            mock_agent_cls.call_args.kwargs["host_context_clause_registry"]
            is registry
        )
        assert future_agent._host_context_publication_gate is publication_gate
        future_agent.defer_agent_readiness_to_host.assert_called_once_with()

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_initializing_turn_rebinds_registry_before_gate_release(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
    ):
        """An unregistered ready-hook turn observes the published host policy."""

        old_registry = object()
        current_registry = object()
        publication_gate = asyncio.Event()
        initialize_started = asyncio.Event()

        class InitializingTurnAgent(TurnLifecycleMixin):
            def __init__(self):
                self.did = "did:initializing"
                self.agent_id = self.did
                self._lock_manager = OrderedLockManager()
                self._host_context_clause_registry = old_registry
                self.observed_registry = None

            def validate_host_context_clause_registry(self, registry):
                return None

            def bind_host_context_clause_registry(self, registry):
                self._host_context_clause_registry = registry

            def defer_agent_readiness_to_host(self):
                self.readiness_owned_by_host = True

            async def initialize(self):
                initialize_started.set()
                async with self._turn_lifecycle():
                    self.observed_registry = self._host_context_clause_registry

            async def shutdown(self):
                return None

        agent = InitializingTurnAgent()
        mock_agent_cls.return_value = agent
        mock_get_did.return_value = agent.did
        manager = AgentManager(base_data_dir=tmp_path)
        manager.bind_host_context_clause_registry(old_registry)
        manager.set_host_context_publication_gate(publication_gate)

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            initialize_task = asyncio.create_task(
                manager._initialize_agent(
                    "initializing",
                    LocalAgentConfig(data_dir=Path("initializing"), port=8801),
                )
            )
            await asyncio.wait_for(initialize_started.wait(), timeout=1)
            await asyncio.sleep(0)
            assert manager.list_agents() == {}

            # Host publication misses ``_agents`` but advances the shared state
            # before releasing every ready-hook turn waiting on the same gate.
            manager.bind_host_context_clause_registry(current_registry)
            publication_gate.set()
            initialized = await asyncio.wait_for(initialize_task, timeout=1)

        assert initialized is agent
        assert agent.observed_registry is current_registry
        assert agent._host_context_publication_generation == 2
        assert manager.list_agents() == {}

    @pytest.mark.asyncio
    async def test_open_gate_cold_agent_readiness_waits_for_onboarding(
        self,
        tmp_path,
    ):
        """A real cold-agent ready pass cannot outrun dynamic registration."""

        manager = AgentManager(base_data_dir=tmp_path)
        publication_gate = asyncio.Event()
        manager.set_host_context_publication_gate(publication_gate)
        publication_gate.set()  # Startup publication/sweep already completed.
        onboarding_done = False
        observations: list[tuple[bool, bool]] = []

        async def onboard(_name, _agent):
            nonlocal onboarding_done
            onboarding_done = True

        manager.set_agent_registration_hook(onboard)
        agent = KestrelAgent(
            "did:open-gate-cold",
            storage_path=str(tmp_path / "cold.db"),
            llm_service=MagicMock(),
        )

        async def ready_hook(_agent):
            async with manager.a2a_execution_lease():
                observations.append(
                    (
                        manager.get_agent("open-gate-cold") is agent,
                        onboarding_done,
                    )
                )

        agent.features["readiness-probe"] = SimpleNamespace(
            name="readiness-probe",
            on_agent_ready=ready_hook,
        )

        async def initialize_through_real_readiness_path():
            await agent._run_or_defer_agent_ready_hooks()

        agent.initialize = initialize_through_real_readiness_path
        agent.shutdown = AsyncMock()

        with (
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
                new=AsyncMock(return_value=agent.did),
            ),
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.KestrelAgent",
                return_value=agent,
            ),
            patch("kestrel_sovereign.multi_agent.agent_manager.LLMService"),
            patch.object(LocalAgentConfig, "validate_runtime", return_value=[]),
        ):
            loaded = await asyncio.wait_for(
                manager.load_agent(
                    "open-gate-cold",
                    LocalAgentConfig(
                        data_dir=Path("open-gate-cold"),
                        port=8801,
                    ),
                ),
                timeout=1,
            )

        assert loaded is agent
        assert observations == [(True, True)]
        assert agent._agent_ready_hooks_completed is True
        assert agent._agent_readiness_host_owned is False
        assert agent._agent_ready_hooks_deferred is False

    @pytest.mark.asyncio
    async def test_late_registration_consumes_deferred_readiness_after_snapshot(self):
        """A cold agent cannot miss the server's one-time readiness sweep."""

        manager = AgentManager()
        publication_gate = asyncio.Event()
        manager.set_host_context_publication_gate(publication_gate)
        publication_gate.set()  # The server snapshot has already completed.

        agent = _make_mock_agent("did:late-ready")
        agent.complete_deferred_agent_readiness = AsyncMock()
        manager._register_agent("late-ready", agent)

        await manager._on_agent_registered("late-ready", agent)
        await manager._complete_registered_agent_readiness(agent)

        assert manager.get_agent("late-ready") is agent
        agent.complete_deferred_agent_readiness.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_server_deferred_readiness_is_bounded_by_signed_expiry(
        self,
        monkeypatch,
    ):
        """One expired child is fenced without aborting healthy peer readiness."""

        manager = AgentManager()
        publication_gate = asyncio.Event()
        manager.set_host_context_publication_gate(publication_gate)
        child_name = "deferred-expiry-child"
        child_did = "did:test:deferred-expiry-child"
        child = _make_mock_agent(child_did)
        child._persisted_spawn_mandate = SpawnMandate(
            parent_did="did:test:deferred-expiry-parent",
            child_did=child_did,
            ttl_seconds=1,
            parent_signature="signed-deferred-expiry",
        )
        effects: list[str] = []

        async def complete_readiness():
            await asyncio.sleep(0.05)
            effects.append("dispatched")

        child.complete_deferred_agent_readiness = AsyncMock(
            side_effect=complete_readiness
        )
        healthy_name = "healthy-deferred-peer"
        healthy_did = "did:test:healthy-deferred-peer"
        healthy = _make_mock_agent(healthy_did)
        healthy.complete_deferred_agent_readiness = AsyncMock()
        manager._agents[child_name] = child
        manager._agent_names[child_did] = child_name
        manager._agents[healthy_name] = healthy
        manager._agent_names[healthy_did] = healthy_name
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager."
            "remaining_spawn_ttl_seconds",
            lambda *_args, **_kwargs: 0.01,
        )
        publication_gate.set()

        await manager.complete_deferred_agent_readiness()

        assert effects == []
        assert manager.get_agent(child_name) is None
        healthy.complete_deferred_agent_readiness.assert_awaited_once_with()
        assert len(manager.init_failures) == 1
        failed_name, failure = manager.init_failures[0]
        assert failed_name == child_name
        assert isinstance(failure, PersistedSpawnMandateExpiredError)
        assert "expired during agent readiness" in str(failure)

    @pytest.mark.asyncio
    async def test_readiness_snapshot_waits_for_registration_onboarding(self):
        """The startup sweep cannot observe a half-onboarded publication."""

        manager = AgentManager()
        publication_gate = asyncio.Event()
        manager.set_host_context_publication_gate(publication_gate)
        publication_gate.set()
        onboarding_started = asyncio.Event()
        release_onboarding = asyncio.Event()
        onboarding_done = False

        async def onboard(_name, _agent):
            nonlocal onboarding_done
            onboarding_started.set()
            await release_onboarding.wait()
            onboarding_done = True

        manager.set_agent_registration_hook(onboard)
        agent = _make_mock_agent("did:registering")

        async def complete_readiness():
            assert onboarding_done is True
            # A feature-ready cognition may execute an A2A tool. The manager
            # must release its publication writer before invoking the hook.
            async with manager.a2a_execution_lease():
                pass

        agent.complete_deferred_agent_readiness = AsyncMock(
            side_effect=complete_readiness
        )

        async def publish_and_onboard():
            async with manager.a2a_lifecycle_lease():
                manager._register_agent("registering", agent)
                await manager._on_agent_registered("registering", agent)
            await manager._complete_registered_agent_readiness(agent)

        registration = asyncio.create_task(publish_and_onboard())
        await asyncio.wait_for(onboarding_started.wait(), timeout=1)

        startup_sweep = asyncio.create_task(
            manager.complete_deferred_agent_readiness()
        )
        await asyncio.sleep(0)
        agent.complete_deferred_agent_readiness.assert_not_awaited()

        release_onboarding.set()
        await asyncio.wait_for(
            asyncio.gather(registration, startup_sweep),
            timeout=1,
        )

        # Registration consumes the deferred hook after onboarding; the
        # serialized startup sweep then observes the same exact-once agent
        # contract and becomes a no-op in production.
        assert agent.complete_deferred_agent_readiness.await_count == 2

    @pytest.mark.asyncio
    async def test_dynamic_registration_releases_writer_before_ready_cognition(
        self,
        tmp_path,
    ):
        """The real load path lets a ready-hook turn acquire an A2A reader."""

        manager = AgentManager(base_data_dir=tmp_path)
        publication_gate = asyncio.Event()
        manager.set_host_context_publication_gate(publication_gate)
        publication_gate.set()
        onboarding_done = False

        async def onboard(_name, _agent):
            nonlocal onboarding_done
            onboarding_done = True

        manager.set_agent_registration_hook(onboard)
        agent = _make_mock_agent("did:dynamic-ready")

        async def complete_readiness():
            assert onboarding_done is True
            async with manager.a2a_execution_lease():
                pass

        agent.complete_deferred_agent_readiness = AsyncMock(
            side_effect=complete_readiness
        )
        manager._initialize_agent = AsyncMock(return_value=agent)

        loaded = await asyncio.wait_for(
            manager.load_agent(
                "dynamic-ready",
                LocalAgentConfig(data_dir=Path("dynamic-ready"), port=8801),
            ),
            timeout=1,
        )

        assert loaded is agent
        assert manager.get_agent("dynamic-ready") is agent
        agent.complete_deferred_agent_readiness.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_dynamic_registration_cancellation_settles_ready_hook(
        self,
        tmp_path,
    ):
        """Caller cancellation cannot strand a published deferred-ready agent."""

        manager = AgentManager(base_data_dir=tmp_path)
        publication_gate = asyncio.Event()
        manager.set_host_context_publication_gate(publication_gate)
        publication_gate.set()
        agent = _make_mock_agent("did:dynamic-cancel")
        readiness_started = asyncio.Event()
        release_readiness = asyncio.Event()
        readiness_done = False

        async def complete_readiness():
            nonlocal readiness_done
            readiness_started.set()
            await release_readiness.wait()
            readiness_done = True

        agent.complete_deferred_agent_readiness = AsyncMock(
            side_effect=complete_readiness
        )
        manager._initialize_agent = AsyncMock(return_value=agent)

        load = asyncio.create_task(
            manager.load_agent(
                "dynamic-cancel",
                LocalAgentConfig(data_dir=Path("dynamic-cancel"), port=8801),
            )
        )
        await asyncio.wait_for(readiness_started.wait(), timeout=1)
        load.cancel()
        await asyncio.sleep(0)
        assert load.done() is False

        release_readiness.set()
        assert await asyncio.wait_for(load, timeout=1) is agent

        assert readiness_done is True
        assert manager.get_agent("dynamic-cancel") is agent
        assert manager._agent_operations == {}

    @pytest.mark.asyncio
    async def test_create_preserves_config_when_ready_hook_cancels(
        self,
        monkeypatch,
        tmp_path,
    ):
        """A best-effort hook cancellation cannot skip the create handoff."""

        manager = AgentManager(base_data_dir=tmp_path)
        publication_gate = asyncio.Event()
        manager.set_host_context_publication_gate(publication_gate)
        publication_gate.set()
        agent = _make_mock_agent("did:dynamic-ready-child-cancel")
        agent.complete_deferred_agent_readiness = AsyncMock(
            side_effect=asyncio.CancelledError("ready child cancelled")
        )
        manager._data_key_custody_conflict = lambda: None
        manager._initialize_agent = AsyncMock(return_value=agent)
        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            AsyncMock(),
        )

        loaded = await manager.create_agent("DynamicReadyChildCancel")

        assert loaded is agent
        assert manager.get_agent("DynamicReadyChildCancel") is agent
        assert manager._created_configs["DynamicReadyChildCancel"] == LocalAgentConfig(
            data_dir=Path("agent_data") / "DynamicReadyChildCancel",
            port=8801,
            autostart=True,
        )
        agent.shutdown.assert_not_awaited()
        assert manager._agent_operations == {}

    @pytest.mark.asyncio
    async def test_create_cancellation_during_readiness_preserves_config_handoff(
        self,
        monkeypatch,
        tmp_path,
    ):
        """A published create remains persistable after caller cancellation."""

        manager = AgentManager(base_data_dir=tmp_path)
        publication_gate = asyncio.Event()
        manager.set_host_context_publication_gate(publication_gate)
        publication_gate.set()
        agent = _make_mock_agent("did:dynamic-create-cancel")
        readiness_started = asyncio.Event()
        release_readiness = asyncio.Event()

        async def complete_readiness():
            readiness_started.set()
            await release_readiness.wait()

        agent.complete_deferred_agent_readiness = AsyncMock(
            side_effect=complete_readiness
        )
        manager._data_key_custody_conflict = lambda: None
        manager._initialize_agent = AsyncMock(return_value=agent)
        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            AsyncMock(),
        )

        create = asyncio.create_task(manager.create_agent("DynamicCreate"))
        await asyncio.wait_for(readiness_started.wait(), timeout=1)
        create.cancel()
        await asyncio.sleep(0)
        assert create.done() is False

        release_readiness.set()
        assert await asyncio.wait_for(create, timeout=1) is agent
        assert manager.get_agent("DynamicCreate") is agent
        assert manager._created_configs["DynamicCreate"] == LocalAgentConfig(
            data_dir=Path("agent_data") / "DynamicCreate",
            port=8801,
            autostart=True,
        )
        assert manager._agent_operations == {}

    @pytest.mark.asyncio
    async def test_load_from_config_initializes_concurrently_and_registers_in_order(self):
        """Slow agents overlap without making fleet/UI order nondeterministic."""
        config = MultiAgentConfig(
            agents={
                "first": LocalAgentConfig(data_dir=Path("/tmp/first"), port=8801),
                "second": LocalAgentConfig(data_dir=Path("/tmp/second"), port=8802),
            }
        )
        manager = AgentManager(base_data_dir=Path("/tmp"))
        both_started = asyncio.Event()
        release = asyncio.Event()
        started = []

        async def initialize(name, _config):
            started.append(name)
            if len(started) == 2:
                both_started.set()
            await release.wait()
            return _make_mock_agent(f"did:{name}")

        manager._initialize_agent = initialize
        load_task = asyncio.create_task(manager.load_from_config(config))

        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert started == ["first", "second"]
        release.set()

        assert await load_task == 2
        assert list(manager._agents) == ["first", "second"]

    @pytest.mark.asyncio
    async def test_failed_batch_identity_read_wakes_signed_sibling(self, tmp_path):
        """A pre-DID initializer failure cannot deadlock its batch siblings."""

        parent_did = "did:test:failed-batch-parent"
        child_did = "did:test:failed-batch-child"
        _parent, mandate = _signed_restored_mandate(parent_did, child_did)
        child = _make_mock_agent(child_did)
        child._persisted_spawn_mandate = mandate
        manager = AgentManager(base_data_dir=tmp_path)
        config = MultiAgentConfig(
            agents={
                "FailedParent": LocalAgentConfig(
                    data_dir=tmp_path / "failed-parent",
                    port=8801,
                ),
                "WaitingChild": LocalAgentConfig(
                    data_dir=tmp_path / "waiting-child",
                    port=8802,
                ),
            }
        )

        async def initialize(name, _config):
            if name == "FailedParent":
                await asyncio.sleep(0)
                raise RuntimeError("anchor read failed")
            await manager._await_admitted_parent_candidate(name, parent_did)
            return child

        manager._initialize_agent = initialize

        assert await asyncio.wait_for(
            manager.load_from_config(config),
            timeout=1.0,
        ) == 0
        assert {name for name, _failure in manager.init_failures} == {
            "FailedParent",
            "WaitingChild",
        }
        child.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_from_config_publishes_signed_parent_before_child(
        self, tmp_path
    ):
        parent_did = "did:test:batch-parent"
        child_did = "did:test:batch-child"
        parent, mandate = _signed_restored_mandate(
            parent_did,
            child_did,
            ttl_seconds=0,
        )
        child = _make_mock_agent(child_did)
        child._persisted_spawn_mandate = mandate
        config = MultiAgentConfig(
            agents={
                "ChildFirstInConfig": LocalAgentConfig(
                    data_dir=tmp_path / "child", port=8801
                ),
                "ParentSecondInConfig": LocalAgentConfig(
                    data_dir=tmp_path / "parent", port=8802
                ),
            }
        )
        manager = AgentManager(base_data_dir=tmp_path)
        initialized = {
            "ChildFirstInConfig": child,
            "ParentSecondInConfig": parent,
        }
        manager._initialize_agent = AsyncMock(
            side_effect=lambda name, _config: initialized[name]
        )

        assert await manager.load_from_config(config) == 2
        assert list(manager._agents) == [
            "ParentSecondInConfig",
            "ChildFirstInConfig",
        ]
        assert manager.get_children(parent_did) == ["ChildFirstInConfig"]

    @pytest.mark.asyncio
    async def test_batch_stages_parent_identity_before_child_active_boot(
        self, tmp_path
    ):
        """A child-first batch works even when one active boot slot is available."""

        parent_did = "did:test:staged-parent"
        child_did = "did:test:staged-child"
        parent_template, mandate = _signed_restored_mandate(
            parent_did,
            child_did,
            ttl_seconds=0,
        )
        constructed: dict[str, object] = {}

        class HostedCandidate:
            def __init__(self, *, did, **_kwargs):
                self.did = did
                self.agent_id = did
                self.features = {}
                self.identity = (
                    parent_template.identity if did == parent_did else None
                )
                self._private_key = (
                    parent_template._private_key if did == parent_did else None
                )
                self._persisted_spawn_mandate = (
                    mandate if did == child_did else None
                )
                constructed[did] = self

            async def initialize(self):
                await self._host_authority_preflight(self)

            async def run_agent_ready_hooks(self):
                return None

            async def shutdown(self):
                return None

        config = MultiAgentConfig(
            agents={
                "Child": LocalAgentConfig(
                    data_dir=tmp_path / "child", port=8801
                ),
                "Parent": LocalAgentConfig(
                    data_dir=tmp_path / "parent", port=8802
                ),
            }
        )
        manager = AgentManager(base_data_dir=tmp_path)
        manager._init_concurrency = 1

        async def read_did(path, **_kwargs):
            return child_did if "child" in str(path) else parent_did

        with (
            patch.object(LocalAgentConfig, "validate_runtime", return_value=[]),
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
                side_effect=read_did,
            ),
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.KestrelAgent",
                HostedCandidate,
            ),
        ):
            loaded = await manager.load_from_config(config)

        assert loaded == 2
        assert set(constructed) == {parent_did, child_did}
        assert list(manager._agents) == ["Parent", "Child"]

    @pytest.mark.asyncio
    async def test_batch_verifies_staged_parent_before_child_active_boot(
        self,
        tmp_path,
    ):
        """A child cannot cross boot through an invalid staged parent mandate."""

        grand_did = "did:test:staged-grand"
        parent_did = "did:test:staged-leaf-parent"
        child_did = "did:test:staged-rejected-child"
        grand, parent_mandate = _signed_restored_mandate(
            grand_did,
            parent_did,
            max_child_depth=2,
        )
        parent, child_mandate = _signed_restored_mandate(
            parent_did,
            child_did,
            max_child_depth=1,
        )
        parent_mandate = replace(
            parent_mandate,
            purpose="tampered after signing",
        )
        templates = {
            grand_did: grand,
            parent_did: parent,
        }
        mandates = {
            grand_did: None,
            parent_did: parent_mandate,
            child_did: child_mandate,
        }
        active_boot_crossings: list[str] = []

        class HostedCandidate:
            def __init__(self, *, did, **_kwargs):
                template = templates.get(did)
                self.did = did
                self.agent_id = did
                self.features = {}
                self.identity = (
                    vars(template).get("identity") if template is not None else None
                )
                self._private_key = (
                    vars(template).get("_private_key")
                    if template is not None
                    else None
                )
                self._persisted_spawn_mandate = None

            async def initialize(self):
                # Model storage_privacy freezing each durable receipt before
                # AgentManager's hosted authority preflight.
                self._persisted_spawn_mandate = mandates[self.did]
                await self._host_authority_preflight(self)
                active_boot_crossings.append(self.did)

            async def run_agent_ready_hooks(self):
                return None

            async def shutdown(self):
                return None

        config = MultiAgentConfig(
            agents={
                "Child": LocalAgentConfig(
                    data_dir=tmp_path / "child", port=8801
                ),
                "Parent": LocalAgentConfig(
                    data_dir=tmp_path / "parent", port=8802
                ),
                "Grand": LocalAgentConfig(
                    data_dir=tmp_path / "grand", port=8803
                ),
            }
        )
        manager = AgentManager(base_data_dir=tmp_path)
        manager._init_concurrency = 1

        async def read_did(path, **_kwargs):
            text = str(path)
            if "child" in text:
                return child_did
            if "parent" in text:
                return parent_did
            return grand_did

        with (
            patch.object(LocalAgentConfig, "validate_runtime", return_value=[]),
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
                side_effect=read_did,
            ),
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.KestrelAgent",
                HostedCandidate,
            ),
        ):
            loaded = await manager.load_from_config(config)

        assert loaded == 1
        assert child_did not in active_boot_crossings
        assert list(manager._agents) == ["Grand"]
        assert any(name == "Parent" for name, _error in manager.init_failures)
        assert any(name == "Child" for name, _error in manager.init_failures)

    @pytest.mark.asyncio
    async def test_concurrent_dynamic_child_waits_for_parent_authority_evidence(
        self,
        tmp_path,
    ):
        """A visible initializer is not inferred to be a governing root."""

        parent_did = "did:test:dynamic-evidence-parent"
        child_did = "did:test:dynamic-evidence-child"
        absent_grand_did = "did:test:dynamic-evidence-absent-grand"
        parent, child_mandate = _signed_restored_mandate(
            parent_did,
            child_did,
            ttl_seconds=0,
        )
        unsigned_parent_lineage = SpawnMandate(
            parent_did=absent_grand_did,
            child_did=parent_did,
            ttl_seconds=0,
        )
        release_parent_evidence = asyncio.Event()
        parent_initialize_started = asyncio.Event()
        child_preflight_started = asyncio.Event()
        active_boot_crossings: list[str] = []

        class HostedCandidate:
            def __init__(self, *, did, **_kwargs):
                self.did = did
                self.agent_id = did
                self.features = {}
                self.identity = parent.identity if did == parent_did else None
                self._private_key = (
                    parent._private_key if did == parent_did else None
                )
                self._persisted_spawn_mandate = None

            async def initialize(self):
                if self.did == parent_did:
                    parent_initialize_started.set()
                    await release_parent_evidence.wait()
                    self._persisted_spawn_mandate = unsigned_parent_lineage
                else:
                    self._persisted_spawn_mandate = child_mandate
                    child_preflight_started.set()
                await self._host_authority_preflight(self)
                active_boot_crossings.append(self.did)

            async def run_agent_ready_hooks(self):
                return None

            async def shutdown(self):
                return None

        manager = AgentManager(base_data_dir=tmp_path)

        async def read_did(path, **_kwargs):
            return parent_did if "parent" in str(path) else child_did

        with (
            patch.object(LocalAgentConfig, "validate_runtime", return_value=[]),
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
                side_effect=read_did,
            ),
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.KestrelAgent",
                HostedCandidate,
            ),
        ):
            parent_load = asyncio.create_task(
                manager.load_agent(
                    "DynamicParent",
                    LocalAgentConfig(data_dir=tmp_path / "parent", port=8801),
                )
            )
            await asyncio.wait_for(parent_initialize_started.wait(), timeout=1.0)
            child_load = asyncio.create_task(
                manager.load_agent(
                    "DynamicChild",
                    LocalAgentConfig(data_dir=tmp_path / "child", port=8802),
                )
            )
            await asyncio.wait_for(child_preflight_started.wait(), timeout=1.0)
            await asyncio.sleep(0)

            assert child_did not in active_boot_crossings
            assert child_load.done() is False
            release_parent_evidence.set()
            assert await parent_load is manager.get_agent("DynamicParent")
            with pytest.raises(RuntimeError, match="non-governing parent"):
                await child_load

        assert active_boot_crossings == [parent_did]
        assert manager.get_agent("DynamicChild") is None

    @pytest.mark.asyncio
    async def test_concurrent_dynamic_child_waits_for_parent_identity_read(
        self,
        tmp_path,
    ):
        """An admitted parent cannot be missed before its constructor is staged."""

        parent_did = "did:test:dynamic-anchor-parent"
        child_did = "did:test:dynamic-anchor-child"
        parent, child_mandate = _signed_restored_mandate(
            parent_did,
            child_did,
            ttl_seconds=0,
        )
        parent_anchor_read = asyncio.Event()
        release_parent_anchor = asyncio.Event()
        child_initialize_started = asyncio.Event()

        class HostedCandidate:
            def __init__(self, *, did, **_kwargs):
                self.did = did
                self.agent_id = did
                self.features = {}
                self.identity = parent.identity if did == parent_did else None
                self._private_key = (
                    parent._private_key if did == parent_did else None
                )
                self._persisted_spawn_mandate = None

            async def initialize(self):
                if self.did == child_did:
                    self._persisted_spawn_mandate = child_mandate
                    child_initialize_started.set()
                await self._host_authority_preflight(self)

            async def run_agent_ready_hooks(self):
                return None

            async def shutdown(self):
                return None

        manager = AgentManager(base_data_dir=tmp_path)

        async def read_did(path, **_kwargs):
            if "parent" in str(path):
                parent_anchor_read.set()
                await release_parent_anchor.wait()
                return parent_did
            return child_did

        with (
            patch.object(LocalAgentConfig, "validate_runtime", return_value=[]),
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
                side_effect=read_did,
            ),
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.KestrelAgent",
                HostedCandidate,
            ),
        ):
            parent_load = asyncio.create_task(
                manager.load_agent(
                    "AnchorParent",
                    LocalAgentConfig(data_dir=tmp_path / "parent", port=8801),
                )
            )
            await asyncio.wait_for(parent_anchor_read.wait(), timeout=1.0)
            child_load = asyncio.create_task(
                manager.load_agent(
                    "AnchorChild",
                    LocalAgentConfig(data_dir=tmp_path / "child", port=8802),
                )
            )
            await asyncio.wait_for(child_initialize_started.wait(), timeout=1.0)
            await asyncio.sleep(0)

            assert child_load.done() is False
            release_parent_anchor.set()
            loaded_parent, loaded_child = await asyncio.gather(
                parent_load,
                child_load,
            )

        assert loaded_parent is manager.get_agent("AnchorParent")
        assert loaded_child is manager.get_agent("AnchorChild")

    @pytest.mark.asyncio
    async def test_dynamic_child_is_cleaned_when_staged_parent_withdraws(
        self,
        tmp_path,
    ):
        """Waiting for parent publication remains inside child cleanup custody."""

        parent_did = "did:test:withdrawing-parent"
        child_did = "did:test:waiting-child"
        parent = _make_mock_agent(parent_did)
        parent._agent_manager_published = False
        parent._agent_manager_publication_event = asyncio.Event()
        child = _make_mock_agent(child_did)
        child._persisted_spawn_mandate = SpawnMandate(
            parent_did=parent_did,
            child_did=child_did,
            ttl_seconds=0,
            parent_signature="signed-for-publication-wait",
        )
        manager = AgentManager(base_data_dir=tmp_path)
        manager._initializing_agents["WithdrawingParent"] = parent

        async def initialize(name, _config, **_kwargs):
            manager._initializing_agents[name] = child
            return child

        manager._initialize_agent = AsyncMock(side_effect=initialize)
        load = asyncio.create_task(
            manager.load_agent(
                "WaitingChild",
                LocalAgentConfig(data_dir=tmp_path / "child", port=8802),
            )
        )
        await asyncio.sleep(0)
        assert load.done() is False

        manager._initializing_agents.pop("WithdrawingParent")
        parent._agent_manager_publication_event.set()

        with pytest.raises(
            PersistedSpawnParentUnavailableError,
            match="withdrew before routing publication",
        ):
            await load

        assert "WaitingChild" not in manager._initializing_agents
        assert manager._preflight_spawn_reservations == {}
        child.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failed_publication_rolls_back_host_routes_assets_and_mode(
        self, tmp_path
    ):
        """A rejected private candidate leaves no app-level onboarding state."""

        from kestrel_sovereign import server
        from kestrel_sovereign.features.base import UIContributions

        static_dir = tmp_path / "feature-static"
        static_dir.mkdir()
        (static_dir / "panel.js").write_text("export const panel = true;\n")
        router = APIRouter()

        @router.get("/rejected-candidate-route")
        async def rejected_candidate_route():
            return {"leaked": True}

        feature = SimpleNamespace(
            name="RejectedFeature",
            enabled=True,
            receiver=None,
            get_router=lambda: router,
            get_ui_contributions=lambda: UIContributions(
                modules=["panel.js"],
                static_dir=str(static_dir),
                capability="rejected",
            ),
        )
        agent = _make_mock_agent("did:test:rejected-onboarding")
        agent.features = {"RejectedFeature": feature}
        agent.peer_directory_router = None
        agent.peer_requester = None
        manager = AgentManager(base_data_dir=tmp_path)
        manager._initialize_agent = AsyncMock(return_value=agent)
        app = FastAPI()
        app.state.agent_manager = manager
        app.state.agent = None
        app.state.demo_mode = "prior-mode"
        route_ids_before = [id(route) for route in app.routes]
        manager.set_agent_registration_hook(
            lambda name, candidate: server._onboard_host_registered_agent(
                app, manager, name, candidate
            )
        )
        manager._register_agent = MagicMock(
            side_effect=RuntimeError("publication rejected")
        )

        with pytest.raises(RuntimeError, match="publication rejected"):
            await manager.load_agent(
                "Rejected",
                LocalAgentConfig(data_dir="unused", port=8801),
            )

        assert [id(route) for route in app.routes] == route_ids_before
        assert app.state.demo_mode == "prior-mode"
        assert getattr(app.state, "_feature_routes", None) is None
        assert getattr(app.state, "_feature_ui_mounts", None) is None
        assert manager.get_agent("Rejected") is None

    @pytest.mark.asyncio
    async def test_failed_onboarding_rollback_cannot_erase_later_registration(
        self, tmp_path,
    ):
        """App snapshot rollback remains inside the registration writer."""

        from kestrel_sovereign import server
        from kestrel_sovereign.features.base import UIContributions

        static_dir = tmp_path / "concurrent-feature-static"
        static_dir.mkdir()
        (static_dir / "panel.js").write_text("export const panel = true;\n")

        def candidate(agent_id: str, route_path: str):
            router = APIRouter()

            @router.get(route_path)
            async def route():
                return {"agent": agent_id}

            feature = SimpleNamespace(
                name=f"Feature-{agent_id}",
                enabled=True,
                receiver=None,
                get_router=lambda: router,
                get_ui_contributions=lambda: UIContributions(
                    modules=["panel.js"],
                    static_dir=str(static_dir),
                    capability=agent_id,
                ),
            )
            agent = _make_mock_agent(agent_id)
            agent.features = {feature.name: feature}
            agent.peer_directory_router = None
            agent.peer_requester = None
            return agent

        agents = {
            "First": candidate("did:test:first", "/first-candidate-route"),
            "Second": candidate("did:test:second", "/second-candidate-route"),
        }
        manager = AgentManager(base_data_dir=tmp_path)

        async def initialize(name, _config, **_kwargs):
            return agents[name]

        manager._initialize_agent = initialize
        app = FastAPI()
        app.state.agent_manager = manager
        app.state.agent = None
        app.state.demo_mode = "prior-mode"
        rollback_started = asyncio.Event()
        allow_rollback = asyncio.Event()
        second_onboarded = asyncio.Event()

        async def onboard(name, agent):
            rollback = await server._onboard_host_registered_agent(
                app, manager, name, agent
            )
            if name == "Second":
                second_onboarded.set()
                return rollback

            async def paused_rollback():
                rollback_started.set()
                await allow_rollback.wait()
                await rollback()

            return paused_rollback

        manager.set_agent_registration_hook(onboard)
        register = manager._register_agent

        def reject_first(name, agent, **kwargs):
            if name == "First":
                raise RuntimeError("first publication rejected")
            return register(name, agent, **kwargs)

        manager._register_agent = reject_first
        first = asyncio.create_task(
            manager.load_agent(
                "First", LocalAgentConfig(data_dir="first", port=8801)
            )
        )
        await asyncio.wait_for(rollback_started.wait(), timeout=1.0)
        second = asyncio.create_task(
            manager.load_agent(
                "Second", LocalAgentConfig(data_dir="second", port=8802)
            )
        )
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(second_onboarded.wait(), timeout=0.1)
        finally:
            allow_rollback.set()

        with pytest.raises(RuntimeError, match="first publication rejected"):
            await first
        assert await second is agents["Second"]
        assert manager.get_agent("Second") is agents["Second"]
        assert any(
            getattr(route, "path", None) == "/second-candidate-route"
            for route in app.routes
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("router_prefix", ["", "/prefixed"])
    async def test_readiness_rollback_removes_only_its_own_host_routes(
        self, tmp_path, router_prefix,
    ):
        """A late ready failure cannot erase a later committed registration."""

        from kestrel_sovereign import server

        def candidate(agent_id: str, route_path: str):
            router = APIRouter(prefix=router_prefix)

            @router.get(route_path)
            async def route():
                return {"agent": agent_id}

            feature = SimpleNamespace(
                name="SharedReadyFeature",
                enabled=True,
                receiver=None,
                get_router=lambda: router,
                get_ui_contributions=lambda: None,
            )
            agent = _make_mock_agent(agent_id)
            agent.features = {feature.name: feature}
            agent.peer_directory_router = None
            agent.peer_requester = None
            return agent

        agents = {
            "FirstReady": candidate(
                "did:test:first-ready", "/shared-ready-route"
            ),
            "SecondReady": candidate(
                "did:test:second-ready", "/shared-ready-route"
            ),
        }
        manager = AgentManager(base_data_dir=tmp_path)

        async def initialize(name, _config, **_kwargs):
            return agents[name]

        manager._initialize_agent = initialize
        app = FastAPI()
        app.state.agent_manager = manager
        app.state.agent = None
        app.state.demo_mode = "prior-mode"
        first_entered_readiness = asyncio.Event()
        fail_first_readiness = asyncio.Event()

        async def ready(agent):
            if agent is agents["FirstReady"]:
                first_entered_readiness.set()
                await fail_first_readiness.wait()
                raise RuntimeError("first readiness failed")

        manager._run_hosted_agent_ready_hooks = AsyncMock(side_effect=ready)
        manager.set_agent_registration_hook(
            lambda name, agent: server._onboard_host_registered_agent(
                app, manager, name, agent
            )
        )

        first = asyncio.create_task(
            manager.load_agent(
                "FirstReady",
                LocalAgentConfig(data_dir="first-ready", port=8801),
            )
        )
        await asyncio.wait_for(first_entered_readiness.wait(), timeout=1.0)
        second = await manager.load_agent(
            "SecondReady",
            LocalAgentConfig(data_dir="second-ready", port=8802),
        )
        assert second is agents["SecondReady"]
        fail_first_readiness.set()

        with pytest.raises(RuntimeError, match="first readiness failed"):
            await first

        assert manager.get_agent("FirstReady") is None
        assert manager.get_agent("SecondReady") is agents["SecondReady"]
        second_feature = agents["SecondReady"].features["SharedReadyFeature"]
        second_router = second_feature.get_router()
        assert server._resolve_live_route_agent(
            app,
            {"state": {}},
            agents["FirstReady"],
            "SharedReadyFeature",
            server._feature_router_route_selector(second_router, 0),
        ) is agents["SecondReady"]
        expected_path = f"{router_prefix}/shared-ready-route"
        assert sum(
            getattr(route, "path", None) == expected_path
            for route in app.routes
        ) == 1
        with TestClient(app) as client:
            response = client.get(expected_path)
        assert response.status_code == 200
        assert response.json() == {"agent": "did:test:second-ready"}


    def test_preflight_reserves_spawn_cap_before_projection(self):
        """Two concurrent cold children cannot both pass a one-slot cap."""

        parent_did = "did:test:reservation-parent"
        parent, first_mandate = _signed_restored_mandate(
            parent_did,
            "did:test:reservation-child-one",
            ttl_seconds=0,
        )
        second_mandate = sign_mandate(
            SpawnMandate(
                parent_did=parent_did,
                child_did="did:test:reservation-child-two",
                ttl_seconds=0,
            ),
            parent._private_key,
        )
        first = _make_mock_agent(first_mandate.child_did)
        first._persisted_spawn_mandate = first_mandate
        second = _make_mock_agent(second_mandate.child_did)
        second._persisted_spawn_mandate = second_mandate
        manager = AgentManager()
        manager._max_spawned_agents = 1
        manager._register_agent("Parent", parent)

        manager._verify_agent_authority("First", first)
        with pytest.raises(RuntimeError, match="spawned-agent cap"):
            manager._verify_agent_authority("Second", second)

        manager._withdraw_initialized_agent("First", first)
        manager._verify_agent_authority("Second", second)
        assert len(manager._preflight_spawn_reservations) == 1

    @pytest.mark.asyncio
    async def test_batch_projects_authority_before_ready_and_rolls_it_back(
        self, tmp_path
    ):
        parent_did = "did:test:batch-prepared-parent"
        child_did = "did:test:batch-prepared-child"
        parent, mandate = _signed_restored_mandate(parent_did, child_did)
        child = _make_mock_agent(child_did)
        child._persisted_spawn_mandate = mandate
        manager = AgentManager(base_data_dir=tmp_path)
        manager._register_agent("BatchPreparedParent", parent)
        manager._initialize_agent = AsyncMock(return_value=child)
        observed_authority = []
        host_state: list[str] = []
        host_rollback = AsyncMock(
            side_effect=lambda: host_state.remove("BatchPreparedChild")
        )

        async def onboard(name, _agent):
            host_state.append(name)
            return host_rollback

        manager.set_agent_registration_hook(onboard)

        async def fail_after_observing_authority(_agent):
            observed_authority.append(
                manager.get_mandate("BatchPreparedChild")
            )
            raise RuntimeError("batch ready hook failed")

        manager._run_hosted_agent_ready_hooks = AsyncMock(
            side_effect=fail_after_observing_authority
        )
        config = MultiAgentConfig(
            agents={
                "BatchPreparedChild": LocalAgentConfig(
                    data_dir=tmp_path / "child",
                    port=8801,
                )
            }
        )

        assert await manager.load_from_config(config) == 0
        assert observed_authority == [mandate]
        assert manager.get_agent("BatchPreparedChild") is None
        assert manager.get_mandate("BatchPreparedChild") is None
        assert manager.get_children(parent_did) == []
        assert host_state == []
        host_rollback.assert_awaited_once()
        assert "batch ready hook failed" in str(manager.init_failures[0][1])

    @pytest.mark.asyncio
    async def test_batch_ready_hook_is_bounded_by_signed_deadline(self, tmp_path):
        parent_did = "did:test:batch-deadline-parent"
        child_did = "did:test:batch-deadline-child"
        parent, mandate = _signed_restored_mandate(
            parent_did,
            child_did,
            ttl_seconds=2,
            created_at=(
                datetime.now(timezone.utc) - timedelta(seconds=1.4)
            ).isoformat(),
        )
        child = _make_mock_agent(child_did)
        child._persisted_spawn_mandate = mandate
        child_dir = tmp_path / "batch-deadline-child"
        child_dir.mkdir()
        (child_dir / "kestrel_prime.db").touch()
        manager = AgentManager(base_data_dir=tmp_path)
        manager._register_agent("BatchDeadlineParent", parent)
        manager._initialize_agent = AsyncMock(return_value=child)
        wake_effects: list[str] = []

        async def wake_after_expiry(_agent):
            await asyncio.sleep(1.0)
            wake_effects.append("dispatched")

        manager._run_hosted_agent_ready_hooks = AsyncMock(
            side_effect=wake_after_expiry
        )
        config = MultiAgentConfig(
            agents={
                "BatchDeadlineChild": LocalAgentConfig(
                    data_dir=child_dir,
                    port=8801,
                )
            }
        )

        with patch(
            "kestrel_sovereign.multi_agent.agent_manager.remaining_spawn_ttl_seconds",
            return_value=0.5,
        ):
            assert await manager.load_from_config(config) == 0
        assert wake_effects == []
        assert manager.get_agent("BatchDeadlineChild") is None
        assert manager.get_children(parent_did) == []
        assert (child_dir / ".kestrel-spawn-retired").read_text().strip() == child_did
        assert "expired during agent readiness" in str(
            manager.init_failures[0][1]
        )

    @pytest.mark.asyncio
    async def test_load_from_config_withdraws_child_when_parent_is_unavailable(
        self, tmp_path
    ):
        _parent, mandate = _signed_restored_mandate(
            "did:test:disabled-parent",
            "did:test:orphaned-child",
            ttl_seconds=0,
        )
        child = _make_mock_agent("did:test:orphaned-child")
        child._persisted_spawn_mandate = mandate
        config = MultiAgentConfig(
            agents={
                "OrphanedChild": LocalAgentConfig(
                    data_dir=tmp_path / "orphan", port=8801
                )
            }
        )
        manager = AgentManager(base_data_dir=tmp_path)
        manager._initialize_agent = AsyncMock(return_value=child)

        assert await manager.load_from_config(config) == 0
        assert manager.get_agent("OrphanedChild") is None
        assert manager.get_children(mandate.parent_did) == []
        child.shutdown.assert_awaited_once()
        assert manager.init_failures[0][0] == "OrphanedChild"
        assert "parent authority" in str(manager.init_failures[0][1])

    @pytest.mark.asyncio
    async def test_cancelled_startup_shuts_down_completed_unregistered_agent(self):
        config = MultiAgentConfig(
            agents={
                "ready": LocalAgentConfig(data_dir=Path("/tmp/ready"), port=8801),
                "blocked": LocalAgentConfig(data_dir=Path("/tmp/blocked"), port=8802),
            }
        )
        manager = AgentManager(base_data_dir=Path("/tmp"))
        ready_agent = _make_mock_agent("did:ready")
        ready = asyncio.Event()
        block = asyncio.Event()

        async def initialize(name, _config):
            if name == "ready":
                ready.set()
                return ready_agent
            await block.wait()
            return _make_mock_agent("did:blocked")

        manager._initialize_agent = initialize
        load_task = asyncio.create_task(manager.load_from_config(config))
        await asyncio.wait_for(ready.wait(), timeout=1)
        await asyncio.sleep(0)

        load_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await load_task

        ready_agent.shutdown.assert_awaited_once()
        assert manager.list_agents() == {}

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_failed_initializer_shuts_down_partial_agent(
        self, mock_llm_cls, mock_agent_cls, mock_get_did
    ):
        mock_get_did.return_value = "did:partial"
        partial = _make_mock_agent("did:partial")
        partial.initialize.side_effect = RuntimeError("init failed")
        mock_agent_cls.return_value = partial
        manager = AgentManager(base_data_dir=Path("/tmp"))
        config = LocalAgentConfig(data_dir=Path("/tmp/partial"), port=8801)

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            with pytest.raises(RuntimeError, match="init failed"):
                await manager._initialize_agent("partial", config)

        mock_get_did.assert_awaited_once_with(
            str(Path("/tmp/partial").resolve()),
            mode=AgentDIDLookupMode.INITIALIZATION,
        )
        partial.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_in_process_agent_receives_resolved_identity_export_override(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
    ):
        mock_get_did.return_value = "did:claw"
        mock_agent = _make_mock_agent("did:claw")
        mock_agent_cls.return_value = mock_agent
        manager = AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(
            data_dir=Path("agent_data/claw"),
            identity_export_dir=Path("continuity"),
            port=8801,
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("claw", config)

        assert mock_agent_cls.call_args.kwargs["identity_export_dir"] == (
            tmp_path / "agent_data" / "claw" / "continuity"
        ).resolve()

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_hosted_telegram_resolver_is_bound_before_agent_initialize(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
    ):
        """A host resolver is selected by local agent scope, never by input."""

        mock_get_did.return_value = "did:telegram:one"
        mock_agent_cls.return_value = _make_mock_agent("did:telegram:one")
        resolver = object()
        resolver_factory = MagicMock(return_value=resolver)
        manager = AgentManager(
            base_data_dir=tmp_path,
            hosted_telegram_route_attestation_resolver_factory=resolver_factory,
        )
        config = LocalAgentConfig(data_dir=Path("agent_data/telegram"), port=8801)

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("telegram", config)

        resolver_factory.assert_called_once_with(
            "telegram", "did:telegram:one", config
        )
        assert (
            mock_agent_cls.call_args.kwargs[
                "hosted_telegram_route_attestation_resolver"
            ]
            is resolver
        )
        mock_agent_cls.return_value.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_each_agent_llm_service_is_bound_to_its_own_data_dir(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
        monkeypatch,
    ):
        """Per-agent usage rows must follow the agent, not the process env.

        Asserts the resolved directory itself, not merely that the keyword was
        passed: ``LLMService(agent_data_dir=None)`` would keep the keyword while
        silently falling back to ``KESTREL_DB_PATH`` and reinstating #2769.
        """
        monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path / "agent_data" / "claw"))
        mock_agent_cls.side_effect = lambda **kw: _make_mock_agent(kw["did"])

        manager = AgentManager(base_data_dir=tmp_path)
        seen = {}
        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            for name in ("claw", "meridian"):
                mock_get_did.return_value = f"did:{name}"
                await manager._initialize_agent(
                    name,
                    LocalAgentConfig(data_dir=Path("agent_data") / name, port=8801),
                )
                seen[name] = mock_llm_cls.call_args.kwargs["agent_data_dir"]

        for name, passed in seen.items():
            assert passed == (tmp_path / "agent_data" / name).resolve()

        # Two agents in one process must not collapse onto one usage DB — the
        # exact condition that produced the live misattribution.
        assert seen["claw"] != seen["meridian"]

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_in_process_agent_receives_semantic_inference_profile(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
    ):
        """The managed config, not a sidecar TOML, controls nightly closure."""
        mock_get_did.return_value = "did:claw"
        mock_agent_cls.return_value = _make_mock_agent("did:claw")
        manager = AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(
            data_dir=Path("agent_data/claw"),
            port=8801,
            semantic_inference={
                "enabled": True,
                "rdfs_version": "1.0.0",
                "ontology": {
                    "namespace": "http://www.w3.org/2000/01/rdf-schema#",
                    "version": "1.0.0",
                    "content_digest": "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
                    "compatibility_profile": "semantic-kb-v1",
                },
                "limits": {"max_source_assertions": 17},
            },
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("claw", config)

        profile = mock_agent_cls.call_args.kwargs["semantic_inference_profile"]
        assert profile == InferenceProfile(
            OntologyRef(
                "http://www.w3.org/2000/01/rdf-schema#",
                "1.0.0",
                "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
                "semantic-kb-v1",
            ),
            "1.0.0",
        )
        assert mock_agent_cls.call_args.kwargs[
            "semantic_inference_limits"
        ].max_source_assertions == 17
        assert mock_agent_cls.call_args.kwargs["semantic_inference_configured"] is True

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_in_process_agent_receives_exact_semantic_capability_selection(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
    ):
        mock_get_did.return_value = "did:claw"
        mock_agent_cls.return_value = _make_mock_agent("did:claw")
        manager = AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(
            data_dir=Path("agent_data/claw"),
            port=8801,
            semantic_capabilities={
                "mode": "experimental",
                "rdf12": {
                    "capability": "rdf-profile:rdf12-cr-20260407-experimental",
                    "version": "0.1.0",
                },
                "sparql12": {
                    "capability": "query-profile:sparql12-20260605-experimental",
                    "version": "0.1.0",
                },
                "shacl12": {
                    "capability": "validation-profile:shacl12-core-20260602-experimental",
                    "version": "0.1.0",
                },
                "shape_set": {
                    "identifier": "kestrel-assertion-shapes-shacl12-experimental",
                    "version": "0.1.0",
                },
            },
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("claw", config)

        selected = mock_agent_cls.call_args.kwargs["semantic_capabilities"]
        assert selected.allow_experimental is True
        assert selected.shape_set.identifier == "kestrel-assertion-shapes-shacl12-experimental"
        assert mock_agent_cls.call_args.kwargs["semantic_capabilities_configured"] is True

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_in_process_agent_rejects_unavailable_pinned_inference_profile(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
    ):
        """Rule resources are validated before the agent can publish ready."""
        mock_get_did.return_value = "did:claw"
        manager = AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(
            data_dir=Path("agent_data/claw"),
            port=8801,
            semantic_inference={
                "enabled": True,
                "rdfs_version": "9.9.9",
                "ontology": {
                    "namespace": "http://www.w3.org/2000/01/rdf-schema#",
                    "version": "1.0.0",
                    "content_digest": "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
                    "compatibility_profile": "semantic-kb-v1",
                },
            },
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            with pytest.raises(InferenceError, match="RDFS"):
                await manager._initialize_agent("claw", config)

        mock_agent_cls.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_sqlite_agent_keeps_storage_derived_isolated_runtime_layout(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
    ):
        mock_get_did.return_value = "did:claw"
        mock_agent_cls.return_value = _make_mock_agent("did:claw")
        manager = AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(
            data_dir=Path("agent_data/claw"),
            port=8801,
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("claw", config)

        agent_root = (tmp_path / "agent_data" / "claw").resolve()
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["identity_export_dir"] == agent_root
        assert "isolated_runtime_root" not in kwargs
        assert "isolated_runtime_namespace" not in kwargs
        assert "isolated_runtime_hosted" not in kwargs

    @pytest.mark.asyncio
    async def test_postgres_factory_constructs_real_agent_with_derived_namespace(
        self, monkeypatch, tmp_path
    ):
        """Exercise the real derive -> KestrelAgent validation seam."""
        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://host/kestrel")
        did = "did:kestrel:real-hosted-construction"
        llm_service = MagicMock()
        llm_service.providers = []
        manager = AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(data_dir=Path("agent_data/real"), port=8801)
        (tmp_path / "agent_data" / "real").mkdir(parents=True)

        with patch.object(
            LocalAgentConfig, "validate_runtime", return_value=[]
        ), patch(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            new=AsyncMock(return_value=did),
        ), patch(
            "kestrel_sovereign.multi_agent.agent_manager.LLMService",
            return_value=llm_service,
        ), patch.object(
            KestrelAgent, "initialize", new=AsyncMock()
        ):
            agent = await manager._initialize_agent("Real", config)

        expected_namespace = derive_isolated_runtime_namespace(did)
        assert type(agent) is KestrelAgent
        assert len(expected_namespace) <= 64
        assert str(agent.isolated_runtime_namespace) == expected_namespace
        assert agent.isolated_runtime_path == (
            tmp_path / "isolated_feature_runtime" / expected_namespace
        ).resolve()
        assert agent.isolated_runtime_legacy_root == (
            tmp_path / "agent_data" / "real" / "feature_venvs"
        ).resolve()

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_postgres_host_factory_supplies_distinct_isolated_runtime_scopes(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        monkeypatch,
        tmp_path,
    ):
        """The real in-process factory owns scoped mutable feature state."""
        agent_a = _make_mock_agent("did:tenant-a")
        agent_b = _make_mock_agent("did:tenant-b")
        mock_agent_cls.side_effect = [agent_a, agent_b]
        mock_get_did.side_effect = ["did:tenant-a", "did:tenant-b"]
        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://host/kestrel")
        manager = AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(data_dir=Path("agent_data/companion"), port=8801)

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("Companion A", config)
            await manager._initialize_agent("Companion B", config)

        first, second = (call.kwargs for call in mock_agent_cls.call_args_list)
        expected_root = (tmp_path / "isolated_feature_runtime").resolve()
        assert first["isolated_runtime_root"] == expected_root
        assert second["isolated_runtime_root"] == expected_root
        assert first["isolated_runtime_hosted"] is True
        assert second["isolated_runtime_hosted"] is True
        expected_legacy_root = (
            tmp_path / "agent_data" / "companion" / "feature_venvs"
        ).resolve()
        assert first["isolated_runtime_legacy_root"] == expected_legacy_root
        assert second["isolated_runtime_legacy_root"] == expected_legacy_root
        assert first["isolated_runtime_namespace"] == (
            derive_isolated_runtime_namespace("did:tenant-a")
        )
        assert second["isolated_runtime_namespace"] == (
            derive_isolated_runtime_namespace("did:tenant-b")
        )
        assert first["isolated_runtime_namespace"] != second["isolated_runtime_namespace"]

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_postgres_host_factory_wires_one_shared_pool_into_every_agent(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        monkeypatch,
        tmp_path,
    ):
        from kestrel_sovereign.storage.db.postgres import PostgresBackend

        class _Pool:
            def get_max_size(self):
                return 10

        pool = _Pool()
        host_backend = PostgresBackend.from_pool(
            pool,
            advisory_dsn="postgresql://host/kestrel",
            advisory_max_pool_size=6,
        )
        mock_agent_cls.side_effect = [
            _make_mock_agent("did:tenant-a"),
            _make_mock_agent("did:tenant-b"),
        ]
        mock_get_did.side_effect = ["did:tenant-a", "did:tenant-b"]
        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://host/kestrel")
        manager = AgentManager(
            base_data_dir=tmp_path,
            shared_postgres_backend=host_backend,
        )
        config = LocalAgentConfig(data_dir=Path("agent_data/companion"), port=8801)

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("Companion A", config)
            await manager._initialize_agent("Companion B", config)

        first, second = (call.kwargs for call in mock_agent_cls.call_args_list)
        assert first["pg_pool"] is pool
        assert second["pg_pool"] is pool
        assert first["shared_postgres_advisory_backend"] is host_backend
        assert second["shared_postgres_advisory_backend"] is host_backend
        assert host_backend._advisory_max_pool_size == 6

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_postgres_host_factory_binds_per_agent_lifecycle_policy(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        monkeypatch,
        tmp_path,
    ):
        mock_get_did.return_value = "did:tenant-policy"
        mock_agent_cls.return_value = _make_mock_agent("did:tenant-policy")
        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://host/kestrel")
        observer = MagicMock()
        policy = HostedIsolatedRuntimeLifecyclePolicy(
            idle_timeout_seconds=900,
            idle_timeouts={"TelegramFeature": None},
            telemetry_observer=observer,
        )
        policy_factory = MagicMock(return_value=policy)
        manager = AgentManager(
            base_data_dir=tmp_path,
            hosted_isolated_runtime_lifecycle_policy_factory=policy_factory,
        )
        config = LocalAgentConfig(data_dir=Path("agent_data/policy"), port=8801)

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("Policy Agent", config)

        policy_factory.assert_called_once_with(
            "Policy Agent", "did:tenant-policy", config
        )
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["isolated_runtime_idle_timeout_seconds"] == 900
        assert kwargs["isolated_runtime_idle_timeouts"] == {
            "TelegramFeature": None
        }
        assert kwargs["isolated_runtime_telemetry_observer"] is observer

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_load_from_config_skips_non_autostart(
        self, mock_llm_cls, mock_agent_cls, mock_get_did
    ):
        """Agents with autostart=false should be skipped."""
        config = MultiAgentConfig(
            agents={
                "active": LocalAgentConfig(data_dir=Path("/tmp/active"), port=8801, autostart=True),
                "inactive": LocalAgentConfig(data_dir=Path("/tmp/inactive"), port=8802, autostart=False),
            }
        )

        mock_get_did.return_value = "did:active"
        mock_agent = _make_mock_agent("did:active")
        mock_agent_cls.return_value = mock_agent

        manager = AgentManager(base_data_dir=Path("/tmp"))

        # Patch validate_runtime to return no errors
        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            loaded = await manager.load_from_config(config)

        assert loaded == 1
        assert manager.get_agent("active") is mock_agent
        assert manager.get_agent("inactive") is None

    @pytest.mark.asyncio
    async def test_restart_loads_cold_authority_parent_before_active_child(
        self,
        tmp_path,
    ):
        """An active child makes its cold parent a required boot dependency."""

        parent_did = "did:test:cold-restart-parent"
        child_did = "did:test:active-restart-child"
        parent, mandate = _signed_restored_mandate(parent_did, child_did)
        child = _make_mock_agent(child_did)
        child._persisted_spawn_mandate = mandate
        parent_config = LocalAgentConfig(
            data_dir=Path("agent_data") / "ColdParent",
            port=8801,
            autostart=False,
        )
        child_config = LocalAgentConfig(
            data_dir=Path("agent_data") / "ActiveChild",
            port=8802,
            autostart=True,
        )
        config = MultiAgentConfig(
            agents={
                "ColdParent": parent_config,
                "ActiveChild": child_config,
            }
        )
        SpawnAuthorityRegistry(tmp_path).record_active(
            child_name="ActiveChild",
            child_did=child_did,
            mandate=mandate,
            config=child_config,
        )
        manager = AgentManager(base_data_dir=tmp_path)

        async def initialize(name, _config):
            return parent if name == "ColdParent" else child

        manager._initialize_agent = AsyncMock(side_effect=initialize)
        manager._on_agent_registered = AsyncMock()
        manager._run_hosted_agent_ready_hooks = AsyncMock()

        with patch(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            new=AsyncMock(return_value=parent_did),
        ) as read_anchor:
            loaded = await manager.load_from_config(config)

        assert loaded == 2
        assert manager.get_agent("ColdParent") is parent
        assert manager.get_agent("ActiveChild") is child
        assert list(manager.list_agents()) == ["ColdParent", "ActiveChild"]
        assert config.agents["ColdParent"].autostart is False
        read_anchor.assert_awaited_once_with(
            str(parent_config.resolve_data_dir(tmp_path)),
            mode=AgentDIDLookupMode.INSPECTION,
        )

    @pytest.mark.asyncio
    async def test_restart_resolves_rotated_parent_alias_before_cold_boot(
        self,
        tmp_path,
        post_ceremony_material,
    ):
        """A signed successor DID still selects its stable cold parent row."""

        child_did = "did:test:rotated-parent-cold-child"
        parent_config = LocalAgentConfig(
            data_dir=Path("."),
            port=8801,
            autostart=False,
        )
        child_config = LocalAgentConfig(
            data_dir=Path("agent_data") / "RotatedParentChild",
            port=8802,
            autostart=True,
        )
        mandate = SpawnMandate(
            parent_did=post_ceremony_material.new_did,
            child_did=child_did,
            parent_signature="signed-by-rotated-parent",
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name="RotatedParentChild",
            child_did=child_did,
            mandate=mandate,
            config=child_config,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        config = MultiAgentConfig(
            agents={
                "RotatedParent": parent_config,
                "RotatedParentChild": child_config,
            }
        )

        with patch(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            new=AsyncMock(return_value=post_ceremony_material.legacy_did),
        ):
            required = await manager._required_cold_authority_names(
                config,
                authority_roots=None,
            )

        assert required == frozenset({"RotatedParent"})

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_load_from_config_handles_errors(
        self, mock_llm_cls, mock_agent_cls, mock_get_did
    ):
        """Failed agent loads should log error but not crash."""
        config = MultiAgentConfig(
            agents={
                "broken": LocalAgentConfig(data_dir=Path("/tmp/broken"), port=8801, autostart=True),
            }
        )

        # validate_runtime returns errors
        with patch.object(LocalAgentConfig, "validate_runtime", return_value=["missing db"]):
            manager = AgentManager(base_data_dir=Path("/tmp"))
            loaded = await manager.load_from_config(config)

        assert loaded == 0
        assert manager.get_agent("broken") is None

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_load_from_config_records_init_failures(
        self, mock_llm_cls, mock_agent_cls, mock_get_did
    ):
        """Per-agent init failures are exposed via manager.init_failures so
        the lifespan handler can surface them via /health (#377 lifecycle
        hardening for multi-agent boot — codex review v3 followup).
        """
        from kestrel_sovereign.lifecycle_checks import NoLLMProvidersError

        config = MultiAgentConfig(
            agents={
                "good": LocalAgentConfig(data_dir=Path("/tmp/good"), port=8801, autostart=True),
                "muted": LocalAgentConfig(data_dir=Path("/tmp/muted"), port=8802, autostart=True),
            }
        )

        mock_get_did.side_effect = ["did:good", "did:muted"]
        good_agent = _make_mock_agent("did:good")
        muted_agent = _make_mock_agent("did:muted")
        # The muted agent's initialize raises the lifecycle hardening error.
        muted_agent.initialize = AsyncMock(
            side_effect=NoLLMProvidersError("no providers for muted")
        )
        mock_agent_cls.side_effect = [good_agent, muted_agent]

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            manager = AgentManager(base_data_dir=Path("/tmp"))
            loaded = await manager.load_from_config(config)

        assert loaded == 1
        assert manager.get_agent("good") is good_agent
        assert manager.get_agent("muted") is None

        failures = manager.init_failures
        assert len(failures) == 1
        name, exc = failures[0]
        assert name == "muted"
        assert isinstance(exc, NoLLMProvidersError)
        assert "no providers for muted" in str(exc)

    @pytest.mark.asyncio
    async def test_init_failures_resets_on_each_load(self):
        """A fresh load_from_config call clears prior failures."""
        manager = AgentManager(base_data_dir=Path("/tmp"))
        manager._init_failures = [("stale", RuntimeError("from a previous run"))]

        empty_config = MultiAgentConfig(agents={})
        loaded = await manager.load_from_config(empty_config)

        assert loaded == 0
        assert manager.init_failures == []

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_mandatory_failure_never_publishes_partial_agent(
        self, mock_llm_cls, mock_agent_cls, mock_get_did
    ):
        config = MultiAgentConfig(
            agents={
                "secure": LocalAgentConfig(
                    data_dir=Path("/tmp/secure"), port=8801
                ),
                "broken": LocalAgentConfig(
                    data_dir=Path("/tmp/broken"), port=8802
                ),
            }
        )
        mock_get_did.side_effect = ["did:secure", "did:broken"]
        secure = _make_mock_agent("did:secure")
        broken = _make_mock_agent("did:broken")
        failure = MandatoryFeatureReadinessError(
            "SecurityFeature",
            "initialization",
            "could not initialize",
        )
        broken.initialize.side_effect = failure
        mock_agent_cls.side_effect = [secure, broken]

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            manager = AgentManager(base_data_dir=Path("/tmp"))
            loaded = await manager.load_from_config(config)

        assert loaded == 1
        assert manager.list_agents() == {"secure": secure}
        assert manager.get_agent("broken") is None
        broken.shutdown.assert_awaited_once()
        assert manager.init_failures == [("broken", failure)]

    @pytest.mark.asyncio
    async def test_identity_failure_keeps_healthy_peer_but_never_publishes_broken(
        self,
        caplog,
    ):
        """Fleet startup records a sanitized, non-invokable partial state."""
        config = MultiAgentConfig(
            agents={
                "healthy": LocalAgentConfig(
                    data_dir=Path("/tmp/healthy"), port=8801
                ),
                "broken": LocalAgentConfig(
                    data_dir=Path("/tmp/broken"), port=8802
                ),
            }
        )
        manager = AgentManager(base_data_dir=Path("/tmp"))
        healthy = _make_mock_agent("did:healthy")
        failure = IdentityReadinessError(
            "custody",
            cause_type="DecryptionError",
        )

        async def initialize(name, _config):
            if name == "broken":
                raise failure
            return healthy

        manager._initialize_agent = initialize
        loaded = await manager.load_from_config(config)

        assert loaded == 1
        assert manager.list_agents() == {"healthy": healthy}
        assert manager.get_agent("broken") is None
        assert manager.init_failures == [("broken", failure)]
        assert "identity_custody" in caplog.text
        assert "/tmp/broken" not in caplog.text


class TestCreateAgent:
    """Test create_agent (inception + load)."""

    @pytest.mark.asyncio
    async def test_create_agent_duplicate_name_raises(self):
        """Creating an agent with an existing name should raise ValueError."""
        manager = AgentManager()
        mock = _make_mock_agent("did:existing")
        manager._agents["Claw"] = mock
        manager._agent_names["did:existing"] = "Claw"

        with pytest.raises(ValueError, match="already exists"):
            await manager.create_agent("Claw")

    @pytest.mark.asyncio
    async def test_create_agent_duplicate_case_insensitive(self):
        """Duplicate check should be case-insensitive."""
        manager = AgentManager()
        mock = _make_mock_agent("did:existing")
        manager._agents["Claw"] = mock
        manager._agent_names["did:existing"] = "Claw"

        with pytest.raises(ValueError, match="already exists"):
            await manager.create_agent("claw")

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.inception_service.create_kestrel_identity_async", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_create_agent_success(
        self, mock_llm_cls, mock_agent_cls, mock_get_did, mock_inception, tmp_path
    ):
        """create_agent should run inception and load the agent."""
        mock_get_did.return_value = "did:new-agent"
        mock_agent = _make_mock_agent("did:new-agent")
        mock_agent_cls.return_value = mock_agent

        manager = AgentManager(base_data_dir=tmp_path)

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            agent = await manager.create_agent("NewBot")

        mock_inception.assert_awaited_once()
        assert agent is mock_agent
        assert manager.get_agent("NewBot") is mock_agent
        # Fleet-idleness (#F235): a dynamically-created/spawned agent must get
        # the co-hosted-agents provider so its restart requests cannot bypass
        # the whole-fleet idle gate. Resolves live to the manager's agents.
        assert callable(agent._cohosted_agents_provider)
        assert agent in agent._cohosted_agents_provider()

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.inception_service.create_kestrel_identity_async", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_create_agent_passes_parent_did(
        self, mock_llm_cls, mock_agent_cls, mock_get_did, mock_inception, tmp_path
    ):
        """create_agent should forward parent_did to inception service."""
        mock_get_did.return_value = "did:child"
        mock_agent = _make_mock_agent("did:child")
        mock_agent_cls.return_value = mock_agent

        manager = AgentManager(base_data_dir=tmp_path)

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager.create_agent("ChildBot", parent_did="did:parent-abc")

        # Verify inception was called WITH parent_did
        mock_inception.assert_awaited_once()
        call_kwargs = mock_inception.call_args[1]
        assert call_kwargs["parent_did"] == "did:parent-abc"
        assert call_kwargs["agent_name"] == "ChildBot"

    @pytest.mark.asyncio
    async def test_concurrent_same_name_create_is_admitted_once_before_inception(
        self, monkeypatch, tmp_path
    ):
        """One name owner reaches inception; the concurrent loser does not."""

        manager = AgentManager(base_data_dir=tmp_path)
        inception_started = asyncio.Event()
        allow_inception = asyncio.Event()
        inception_calls = 0
        child = _make_mock_agent("did:test:single-create-owner")

        async def inception(**_kwargs):
            nonlocal inception_calls
            inception_calls += 1
            inception_started.set()
            await allow_inception.wait()

        async def initialize(*_args, **_kwargs):
            return child

        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            inception,
        )
        manager._initialize_agent = initialize

        first = asyncio.create_task(manager.create_agent("SameName"))
        await asyncio.wait_for(inception_started.wait(), timeout=1.0)
        with pytest.raises(ValueError, match="already being initialized or created"):
            await manager.create_agent("samename")
        with pytest.raises(ValueError, match="already being initialized or created"):
            await manager.load_agent(
                "SAMENAME", LocalAgentConfig(data_dir="other", port=8802)
            )

        allow_inception.set()
        assert await asyncio.wait_for(first, timeout=1.0) is child
        assert inception_calls == 1
        assert manager.list_agents() == {"SameName": child}
        assert manager.get_agent_name(child.agent_id) == "SameName"
        child.shutdown.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_spawn_denies_auto_discovery_before_inception_can_write_identity(
        self, monkeypatch, tmp_path
    ):
        """A crash after birth cannot restart an unsigned, uncommitted child."""

        manager = AgentManager(base_data_dir=tmp_path)
        parent = _make_mock_agent("did:test:pending-spawn-parent")
        parent._private_key, _ = generate_secp256k1_keypair()
        parent.identity = None
        parent.features = {}
        inception_wrote_identity = asyncio.Event()
        hold_inception = asyncio.Event()
        reserved_caps: list[int | None] = []
        reserve_pending = manager._spawn_authority_registry.reserve_pending

        def capture_shared_cap(**kwargs):
            reserved_caps.append(kwargs.get("max_authority_slots"))
            return reserve_pending(**kwargs)

        monkeypatch.setattr(
            manager._spawn_authority_registry,
            "reserve_pending",
            capture_shared_cap,
        )

        async def inception(*, output_dir, **_kwargs):
            # This is the process-crash boundary: inception has made the child
            # discoverable, but its final DID has not returned to the caller and
            # therefore cannot yet be bound into a signed authority witness.
            (Path(output_dir) / "kestrel_prime.db").touch()
            inception_wrote_identity.set()
            await hold_inception.wait()

        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            inception,
        )
        spawn = asyncio.create_task(
            manager.spawn_agent(
                "CrashWindowChild",
                parent,
                SpawnMandate(parent_did=parent.agent_id),
            )
        )
        await asyncio.wait_for(inception_wrote_identity.wait(), timeout=1.0)

        discovered = MultiAgentConfig.auto_discover(tmp_path / "agent_data")
        assert "CrashWindowChild" not in discovered.agents
        assert reserved_caps == [manager._max_spawned_agents]

        spawn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await spawn
        pending = manager._spawn_authority_registry.pending()
        assert len(pending) == 1
        assert manager._spawn_cap_slots_in_use() == 1
        assert manager._spawn_authority_registry.reap_orphaned_pending_without_birth(
            reservation_ids=(pending[0].reservation_id,),
        ) == (pending[0],)
        assert manager._spawn_cap_slots_in_use() == 0

    @pytest.mark.asyncio
    async def test_spawn_releases_pending_denial_after_clean_inception_failure(
        self, monkeypatch, tmp_path
    ):
        """A reported no-identity failure does not strand the reserved slot."""

        manager = AgentManager(base_data_dir=tmp_path)
        parent = _make_mock_agent("did:test:failed-inception-parent")
        parent._private_key, _ = generate_secp256k1_keypair()
        parent.identity = None
        parent.features = {}

        async def fail_inception(**_kwargs):
            raise RuntimeError("identity was not created")

        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            fail_inception,
        )

        with pytest.raises(ValueError, match="identity was not created"):
            await manager.spawn_agent(
                "FailedInceptionChild",
                parent,
                SpawnMandate(parent_did=parent.agent_id),
            )

        assert manager._spawn_authority_registry.pending() == ()

    @pytest.mark.asyncio
    async def test_spawn_does_not_reserve_existing_cold_identity_as_partial_birth(
        self, monkeypatch, tmp_path
    ):
        """Reject a pre-existing identity before this attempt reserves capacity."""

        child_name = "ExistingColdIdentity"
        child_did = "did:test:existing-cold-identity"
        identity_db = tmp_path / "agent_data" / child_name / "kestrel_prime.db"
        identity_db.parent.mkdir(parents=True)
        with sqlite3.connect(identity_db) as connection:
            connection.execute(
                "CREATE TABLE graph_nodes ("
                "node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, "
                "label TEXT NOT NULL, properties TEXT)"
            )
            connection.execute(
                "INSERT INTO graph_nodes "
                "(node_id, node_type, label, properties) VALUES (?, ?, ?, ?)",
                (child_did, "agent", child_name, "{}"),
            )

        manager = AgentManager(base_data_dir=tmp_path)
        parent = _make_mock_agent("did:test:existing-cold-parent")
        parent._private_key, _ = generate_secp256k1_keypair()
        parent.identity = None
        parent.features = {}
        inception = AsyncMock(side_effect=FileExistsError("identity already exists"))
        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            inception,
        )

        with pytest.raises(ValueError, match="already contains an agent identity"):
            await manager.spawn_agent(
                child_name,
                parent,
                SpawnMandate(parent_did=parent.agent_id),
            )

        inception.assert_not_awaited()
        assert manager._spawn_authority_registry.pending() == ()
        assert manager._spawn_cap_slots_in_use() == 0

    @pytest.mark.asyncio
    async def test_spawn_releases_pending_denial_when_agent_directory_setup_fails(
        self, monkeypatch, tmp_path
    ):
        """A provably pre-inception filesystem failure cannot reserve a slot."""

        manager = AgentManager(base_data_dir=tmp_path)
        child_name = "DirectorySetupFailureChild"
        child_dir = tmp_path / "agent_data" / child_name
        parent = _make_mock_agent("did:test:directory-setup-parent")
        parent._private_key, _ = generate_secp256k1_keypair()
        parent.identity = None
        parent.features = {}
        original_mkdir = Path.mkdir

        def fail_child_directory(path, *args, **kwargs):
            if path == child_dir:
                raise OSError("agent directory is unavailable")
            return original_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail_child_directory)

        with pytest.raises(OSError, match="agent directory is unavailable"):
            await manager.spawn_agent(
                child_name,
                parent,
                SpawnMandate(parent_did=parent.agent_id),
            )

        assert manager._spawn_authority_registry.pending() == ()
        assert manager._spawn_cap_slots_in_use() == 0

    @pytest.mark.asyncio
    async def test_spawn_retains_pending_denial_after_partial_inception_failure(
        self, monkeypatch, tmp_path
    ):
        """An ordinary exception after a database write is crash-ambiguous."""

        manager = AgentManager(base_data_dir=tmp_path)
        parent = _make_mock_agent("did:test:partial-inception-parent")
        parent._private_key, _ = generate_secp256k1_keypair()
        parent.identity = None
        parent.features = {}

        async def partially_fail(*, output_dir, **_kwargs):
            (Path(output_dir) / "kestrel_prime.db").touch()
            raise RuntimeError("failed after durable identity boundary")

        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            partially_fail,
        )

        with pytest.raises(ValueError, match="durable identity boundary"):
            await manager.spawn_agent(
                "PartialInceptionChild",
                parent,
                SpawnMandate(parent_did=parent.agent_id),
            )

        pending = manager._spawn_authority_registry.pending()
        assert len(pending) == 1
        discovered = MultiAgentConfig.auto_discover(tmp_path / "agent_data")
        assert "PartialInceptionChild" not in discovered.agents
        # The failed attempt is terminal even though its durable birth is
        # ambiguous. A same-process recovery pass can therefore claim the
        # producer lock and reap the empty SQLite shell without a host restart.
        assert manager._spawn_authority_registry.reap_orphaned_pending_without_birth(
            reservation_ids=(pending[0].reservation_id,),
        ) == (pending[0],)
        assert manager._spawn_authority_registry.pending() == ()

    @pytest.mark.asyncio
    async def test_spawn_retains_pending_denial_when_birth_inspection_is_unreadable(
        self, monkeypatch, tmp_path
    ):
        """An inspection error is uncertainty, never proof of no birth."""

        manager = AgentManager(base_data_dir=tmp_path)
        child_name = "UnreadableFailedInceptionChild"
        identity_db = tmp_path / "agent_data" / child_name / "kestrel_prime.db"
        parent = _make_mock_agent("did:test:unreadable-inception-parent")
        parent._private_key, _ = generate_secp256k1_keypair()
        parent.identity = None
        parent.features = {}
        original_stat = Path.stat
        original_exists = os.path.exists

        async def fail_after_birth(*, output_dir, **_kwargs):
            (Path(output_dir) / "kestrel_prime.db").touch()
            raise RuntimeError("failed after unreadable durable birth")

        def unreadable_identity(path, *args, **kwargs):
            if path == identity_db:
                raise PermissionError("identity slot is temporarily unreadable")
            return original_stat(path, *args, **kwargs)

        def suppressed_unreadable_exists(path):
            if Path(path) == identity_db:
                return False
            return original_exists(path)

        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            fail_after_birth,
        )
        monkeypatch.setattr(Path, "stat", unreadable_identity)
        monkeypatch.setattr(os.path, "exists", suppressed_unreadable_exists)

        with pytest.raises(ValueError, match="unreadable durable birth"):
            await manager.spawn_agent(
                child_name,
                parent,
                SpawnMandate(parent_did=parent.agent_id),
            )

        assert len(manager._spawn_authority_registry.pending()) == 1

    @pytest.mark.asyncio
    async def test_shutdown_fences_pre_inception_create_without_deleting_identity(
        self, monkeypatch, tmp_path
    ):
        """An old create cannot publish after shutdown or block later reuse."""

        manager = AgentManager(base_data_dir=tmp_path)
        inception_started = asyncio.Event()
        allow_inception = asyncio.Event()
        child = _make_mock_agent("did:test:stale-create")

        async def inception(**_kwargs):
            inception_started.set()
            await allow_inception.wait()

        async def initialize(*_args, **_kwargs):
            return child

        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            inception,
        )
        manager._initialize_agent = initialize

        create = asyncio.create_task(manager.create_agent("BeforeShutdown"))
        await asyncio.wait_for(inception_started.wait(), timeout=1.0)
        shutdown = asyncio.create_task(manager.shutdown_all())
        while not manager._agent_registration_sealed:
            await asyncio.sleep(0)
        # Inception itself is unbounded external work.  Shutdown fences its
        # later publication but does not hang waiting for that ordinary create.
        assert await asyncio.wait_for(shutdown, timeout=1.0) is None

        allow_inception.set()
        with pytest.raises(RuntimeError, match="began before manager shutdown"):
            await asyncio.wait_for(create, timeout=1.0)

        assert manager.list_agents() == {}
        assert manager._created_configs == {}
        child.shutdown.assert_not_awaited()
        # Inception data remains deliberately intact; a canceled/stale create
        # never guesses whether it is safe to delete an identity directory.
        assert (tmp_path / "agent_data" / "BeforeShutdown").is_dir()
        # The stale admission is rejected before it can initialize a runtime
        # around identity data written before shutdown.
        assert manager._agent_operations == {}

        replacement = _make_mock_agent("did:test:post-shutdown-create")

        async def initialize_replacement(*_args, **_kwargs):
            return replacement

        manager._initialize_agent = initialize_replacement
        assert await manager.load_agent(
            "BeforeShutdown", LocalAgentConfig(data_dir="replacement", port=8802)
        ) is replacement
        assert manager.get_agent("BeforeShutdown") is replacement

    @pytest.mark.asyncio
    async def test_concurrent_same_name_load_rejects_before_second_initialization(self):
        """Direct cold loads share the same admission, not just create_agent."""

        manager = AgentManager()
        initialized = asyncio.Event()
        allow_first = asyncio.Event()
        child = _make_mock_agent("did:test:single-load-owner")
        initialize_calls = 0

        async def initialize(*_args, **_kwargs):
            nonlocal initialize_calls
            initialize_calls += 1
            initialized.set()
            await allow_first.wait()
            return child

        manager._initialize_agent = initialize
        config = LocalAgentConfig(data_dir="same", port=8801)
        first = asyncio.create_task(manager.load_agent("SameLoad", config))
        await asyncio.wait_for(initialized.wait(), timeout=1.0)
        with pytest.raises(ValueError, match="already being initialized or created"):
            await manager.load_agent("sameload", config)

        allow_first.set()
        assert await asyncio.wait_for(first, timeout=1.0) is child
        assert initialize_calls == 1
        assert manager.list_agents() == {"SameLoad": child}
        assert manager._agent_names == {child.agent_id: "SameLoad"}

    @pytest.mark.asyncio
    async def test_cancelled_operation_release_waits_for_state_lock(
        self,
    ) -> None:
        """A completed admission cannot be stranded by cancellation in finally."""

        manager = AgentManager()
        admission, owns_admission = await manager._admit_agent_operation(
            "Completed", kind="test"
        )
        assert owns_admission

        await manager._lock.acquire()
        release = asyncio.create_task(manager._release_agent_operation(admission))
        try:
            await asyncio.sleep(0)
            release.cancel()

            # The owned release is blocked on the state lock. Releasing that
            # lock must finish its mutation before the caller's cancellation
            # can propagate.
            manager._lock.release()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(release, timeout=1.0)
        finally:
            if manager._lock.locked():
                manager._lock.release()

        assert manager._agent_operations == {}


class TestSpawnAgent:
    """Test spawn_agent (delegation chain)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_depth", [-1, True, 1.5])
    async def test_spawn_rejects_depth_outside_durable_schema_before_inception(
        self,
        tmp_path,
        invalid_depth,
    ):
        manager = AgentManager(base_data_dir=tmp_path)
        manager._do_spawn = AsyncMock(return_value=_make_mock_agent("did:test:child"))
        parent = _make_mock_agent("did:test:parent")
        mandate = SpawnMandate(
            parent_did=parent.agent_id,
            max_child_depth=invalid_depth,
        )

        with pytest.raises((TypeError, ValueError), match="max_child_depth"):
            await manager.spawn_agent("InvalidDepthChild", parent, mandate)

        manager._do_spawn.assert_not_awaited()

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        new_callable=AsyncMock,
    )
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_spawn_restart_selection_is_durable_before_child_publication(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        mock_inception,
        tmp_path,
    ):
        config_path = tmp_path / "multi_agent.toml"
        MultiAgentConfig(agents={}).save(config_path)
        child = _make_mock_agent("did:test:crash-safe-child")
        child._raw_storage = SimpleNamespace(
            graph=SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
        )
        mock_get_did.return_value = child.agent_id
        mock_agent_cls.return_value = child
        manager = AgentManager(
            base_data_dir=tmp_path,
            startup_config_path=config_path,
        )
        parent = _make_mock_agent("did:test:crash-safe-parent")
        parent._private_key, _ = generate_secp256k1_keypair()
        parent.identity = None
        original_register = manager._register_agent

        def assert_restart_selection_then_publish(name, agent, **kwargs):
            assert name in MultiAgentConfig.from_file(config_path).agents
            witness = manager._spawn_authority_registry.get(agent.agent_id)
            assert witness is not None
            assert witness.child_name == name
            return original_register(name, agent, **kwargs)

        manager._register_agent = assert_restart_selection_then_publish

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager.spawn_agent(
                "CrashSafeChild",
                parent,
                SpawnMandate(parent_did=parent.agent_id),
            )

    def test_host_spawn_witness_refuses_child_with_missing_local_receipt(
        self, tmp_path
    ):
        child_name = "LostReceiptChild"
        child_did = "did:test:lost-receipt-child"
        parent_did = "did:test:lost-receipt-parent"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        registry = {
            "version": 1,
            "records": {
                child_did: {
                    "child_name": child_name,
                    "child_did": child_did,
                    "parent_did": parent_did,
                    "mandate": {
                        "parent_did": parent_did,
                        "child_did": child_did,
                        "constitution_hash": "",
                        "additional_constraints": {},
                        "budget_allocation": 0.0,
                        "ttl_seconds": 3600,
                        "features_allowed": [],
                        "purpose": "",
                        "max_child_depth": 0,
                        "parent_signature": "signed-host-witness",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "config": local.model_dump(mode="json"),
                    "retired": False,
                },
            }
        }
        registry_path = tmp_path / "agent_data" / ".kestrel-spawn-authority.json"
        registry_path.parent.mkdir()
        registry_path.write_text(
            json.dumps(registry),
            encoding="utf-8",
        )
        manager = AgentManager(base_data_dir=tmp_path)
        parent = _make_mock_agent(parent_did)
        child = _make_mock_agent(child_did)
        child._persisted_spawn_mandate = None
        manager._register_agent("LostReceiptParent", parent)

        with pytest.raises(RuntimeError, match="host spawn witness.*local receipt"):
            manager._verify_agent_authority(child_name, child)

    def test_registration_rechecks_host_witness_before_publication(self, tmp_path):
        """No alternate registrar may bypass the hosted authority preflight."""

        child_name = "LostReceiptAtPublication"
        child_did = "did:test:lost-receipt-at-publication"
        parent_did = "did:test:publication-parent"
        mandate = SpawnMandate(
            parent_did=parent_did,
            child_did=child_did,
            ttl_seconds=3600,
            parent_signature="signed-host-witness",
        )
        SpawnAuthorityRegistry(tmp_path).record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=LocalAgentConfig(
                data_dir=Path("agent_data") / child_name,
                port=8802,
            ),
        )
        manager = AgentManager(base_data_dir=tmp_path)
        parent = _make_mock_agent(parent_did)
        child = _make_mock_agent(child_did)
        child._persisted_spawn_mandate = None
        manager._register_agent("PublicationParent", parent)

        with pytest.raises(RuntimeError, match="host spawn witness.*local receipt"):
            manager._register_agent(child_name, child)

        assert manager.get_agent(child_name) is None
        assert child_did not in manager._agent_names

    @pytest.mark.asyncio
    async def test_host_witness_repairs_interrupted_unsigned_receipt_before_boot(
        self,
        tmp_path,
    ):
        """A crash after the host write cannot brick every later restart."""

        child_name = "InterruptedReceiptChild"
        child_did = "did:test:interrupted-receipt-child"
        parent_did = "did:test:interrupted-receipt-parent"
        parent, mandate = _signed_restored_mandate(parent_did, child_did)
        config = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        SpawnAuthorityRegistry(tmp_path).record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=config,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        manager._register_agent("InterruptedReceiptParent", parent)
        repair_started = asyncio.Event()
        allow_repair = asyncio.Event()

        async def persist_repaired_receipt(*_args, **_kwargs):
            repair_started.set()
            await allow_repair.wait()

        graph = SimpleNamespace(
            add_trusted_cross_agent_edge=AsyncMock(
                side_effect=persist_repaired_receipt,
            )
        )
        active_boot_crossings = []
        evidence_visible_during_repair = []

        class HostedChild:
            def __init__(self, *, did, **_kwargs):
                self.did = did
                self.agent_id = did
                self.identity = None
                self.features = {}
                self._raw_storage = SimpleNamespace(graph=graph)
                self._persisted_spawn_mandate = replace(
                    mandate,
                    parent_signature=None,
                )

            async def initialize(self):
                preflight = asyncio.create_task(
                    self._host_authority_preflight(self)
                )
                await asyncio.wait_for(repair_started.wait(), timeout=1.0)
                evidence_visible_during_repair.append(
                    self._agent_manager_authority_evidence_event.is_set()
                )
                allow_repair.set()
                await preflight
                active_boot_crossings.append(
                    (
                        self.agent_id,
                        vars(self).get("_host_authority_boot_deadline_handle")
                        is not None,
                    )
                )

            async def run_agent_ready_hooks(self):
                return None

            async def shutdown(self):
                return None

        with (
            patch.object(LocalAgentConfig, "validate_runtime", return_value=[]),
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
                new=AsyncMock(return_value=child_did),
            ),
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.KestrelAgent",
                HostedChild,
            ),
        ):
            child = await manager.load_agent(child_name, config)

        graph.add_trusted_cross_agent_edge.assert_awaited_once_with(
            child_did,
            parent_did,
            "spawned_by",
            properties=mandate.to_edge_properties(),
        )
        assert child._persisted_spawn_mandate.to_dict() == mandate.to_dict()
        assert evidence_visible_during_repair == [False]
        assert active_boot_crossings == [(child_did, True)]
        assert manager.get_agent(child_name) is child

    @pytest.mark.parametrize("candidate_name", ["BoundChild", "RenamedChild"])
    def test_host_spawn_witness_refuses_replacement_did_in_bound_slot(
        self,
        tmp_path,
        candidate_name,
    ):
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / "BoundChild",
            port=8802,
        )
        private_key, _ = generate_secp256k1_keypair()
        mandate = sign_mandate(
            SpawnMandate(
                parent_did="did:test:bound-parent",
                child_did="did:test:bound-child",
            ),
            private_key,
        )
        SpawnAuthorityRegistry(tmp_path).record_active(
            child_name="BoundChild",
            child_did=mandate.child_did,
            mandate=mandate,
            config=local,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        if candidate_name != "BoundChild":
            manager._created_configs[candidate_name] = local
        replacement = _make_mock_agent("did:test:replacement-child")
        replacement._persisted_spawn_mandate = None

        with pytest.raises(RuntimeError, match="different child DID"):
            manager._verify_agent_authority(candidate_name, replacement)

    def test_pending_spawn_authority_refuses_direct_cold_publication(self, tmp_path):
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / "PendingColdChild",
            port=8802,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.reserve_pending(
            child_name="PendingColdChild",
            parent_did="did:test:pending-cold-parent",
            mandate=SpawnMandate(parent_did="did:test:pending-cold-parent"),
            config=local,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        manager._created_configs["PendingColdChild"] = local
        candidate = _make_mock_agent("did:test:unsigned-pending-cold-child")
        candidate._persisted_spawn_mandate = None

        with pytest.raises(RuntimeError, match="pending spawn authority"):
            manager._verify_agent_authority("PendingColdChild", candidate)

    def test_host_spawn_witness_recovers_missing_startup_selection(self, tmp_path):
        config_path = tmp_path / "multi_agent.toml"
        MultiAgentConfig(agents={}).save(config_path)
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / "RecoveredChild",
            port=8802,
        )
        private_key, _ = generate_secp256k1_keypair()
        mandate = sign_mandate(
            SpawnMandate(
                parent_did="did:test:recovery-parent",
                child_did="did:test:recovery-child",
            ),
            private_key,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        assert registry.path.parent == tmp_path / "agent_data"
        registry.record_active(
            child_name="RecoveredChild",
            child_did=mandate.child_did,
            mandate=mandate,
            config=local,
        )
        manager = AgentManager(
            base_data_dir=tmp_path,
            startup_config_path=config_path,
        )

        reconciled = manager._reconcile_spawn_authority_restart_roster(
            MultiAgentConfig(agents={})
        )

        assert reconciled.agents["RecoveredChild"] == local
        assert MultiAgentConfig.from_file(config_path).agents["RecoveredChild"] == local

    def test_restart_roster_repair_does_not_persist_runtime_host_overrides(
        self,
        tmp_path,
    ):
        """Platform listen overrides stay effective but never rewrite policy."""

        config_path = tmp_path / "multi_agent.toml"
        persisted = MultiAgentConfig(agents={})
        persisted.host.bind = "0.0.0.0"
        persisted.host.port = 8888
        persisted.save(config_path)
        runtime = MultiAgentConfig.from_file(config_path)
        runtime.host.bind = "127.0.0.1"
        runtime.host.port = 9999
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / "RuntimeOverrideChild",
            port=8802,
        )
        child_did = "did:test:runtime-override-child"
        private_key, _ = generate_secp256k1_keypair()
        mandate = sign_mandate(
            SpawnMandate(
                parent_did="did:test:runtime-override-parent",
                child_did=child_did,
            ),
            private_key,
        )
        SpawnAuthorityRegistry(tmp_path).record_active(
            child_name="RuntimeOverrideChild",
            child_did=child_did,
            mandate=mandate,
            config=local,
        )
        manager = AgentManager(
            base_data_dir=tmp_path,
            startup_config_path=config_path,
        )

        reconciled = manager._reconcile_spawn_authority_restart_roster(runtime)

        assert reconciled.host.bind == "127.0.0.1"
        assert reconciled.host.port == 9999
        assert reconciled.agents["RuntimeOverrideChild"] == local
        reloaded = MultiAgentConfig.from_file(config_path)
        assert reloaded.host.bind == "0.0.0.0"
        assert reloaded.host.port == 8888
        assert reloaded.agents["RuntimeOverrideChild"] == local

    def test_restart_roster_repair_preserves_concurrent_operator_agent_edits(
        self,
        tmp_path,
    ):
        """Recovery merges into the latest file rather than a stale startup model."""

        config_path = tmp_path / "multi_agent.toml"
        stale_runtime = MultiAgentConfig(agents={})
        stale_runtime.host.port = 9999
        MultiAgentConfig(agents={}).save(config_path)
        operator_agent = LocalAgentConfig(
            data_dir=Path("agent_data") / "OperatorAdded",
            port=8803,
            autostart=False,
        )
        latest = MultiAgentConfig(agents={"OperatorAdded": operator_agent})
        latest.host.port = 8888
        latest.save(config_path)
        recovered = LocalAgentConfig(
            data_dir=Path("agent_data") / "RecoveredAlongsideEdit",
            port=8802,
        )
        child_did = "did:test:recovered-alongside-edit"
        private_key, _ = generate_secp256k1_keypair()
        mandate = sign_mandate(
            SpawnMandate(
                parent_did="did:test:recovered-alongside-edit-parent",
                child_did=child_did,
            ),
            private_key,
        )
        SpawnAuthorityRegistry(tmp_path).record_active(
            child_name="RecoveredAlongsideEdit",
            child_did=child_did,
            mandate=mandate,
            config=recovered,
        )
        manager = AgentManager(
            base_data_dir=tmp_path,
            startup_config_path=config_path,
        )

        reconciled = manager._reconcile_spawn_authority_restart_roster(stale_runtime)

        assert reconciled.host.port == 9999
        assert reconciled.agents == {
            "OperatorAdded": operator_agent,
            "RecoveredAlongsideEdit": recovered,
        }
        persisted = MultiAgentConfig.from_file(config_path)
        assert persisted.host.port == 8888
        assert persisted.agents == reconciled.agents

    def test_pending_spawn_authority_removes_explicit_restart_selection(
        self, tmp_path
    ):
        """A configured roster cannot bypass the pre-inception crash fence."""

        local = LocalAgentConfig(
            data_dir=Path("agent_data") / "PendingChild",
            port=8802,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.reserve_pending(
            child_name="PendingChild",
            parent_did="did:test:pending-parent",
            mandate=SpawnMandate(parent_did="did:test:pending-parent"),
            config=local,
        )
        manager = AgentManager(base_data_dir=tmp_path)

        reconciled = manager._reconcile_spawn_authority_restart_roster(
            MultiAgentConfig(agents={"PendingChild": local})
        )

        assert reconciled.agents == {}

    def test_restart_reaps_ownerless_pending_reservation_without_birth(
        self, tmp_path
    ):
        """A crashed pre-inception owner cannot reserve its slot forever."""

        child_name = "OrphanedPendingChild"
        parent_did = "did:test:orphaned-pending-parent"
        reservation_id = "orphaned-reservation"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        proposal = SpawnMandate(parent_did=parent_did)
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.path.parent.mkdir(parents=True, exist_ok=True)
        registry.path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "records": {},
                    "pending": {
                        reservation_id: {
                            "reservation_id": reservation_id,
                            "child_name": child_name,
                            "parent_did": parent_did,
                            "mandate": proposal.to_dict(),
                            "config": local.model_dump(mode="json"),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        manager = AgentManager(base_data_dir=tmp_path)

        reconciled = manager._reconcile_spawn_authority_restart_roster(
            MultiAgentConfig(agents={})
        )

        assert reconciled.agents == {}
        assert SpawnAuthorityRegistry(tmp_path).pending() == ()

    def test_scoped_restart_reaps_ownerless_pending_from_sibling_root(
        self, tmp_path
    ):
        """Root scoping cannot hide an orphan from the host-global spawn cap."""

        child_name = "OrphanedSiblingPendingChild"
        parent_did = "did:test:orphaned-sibling-parent"
        reservation_id = "orphaned-sibling-reservation"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.path.parent.mkdir(parents=True, exist_ok=True)
        registry.path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "records": {},
                    "pending": {
                        reservation_id: {
                            "reservation_id": reservation_id,
                            "child_name": child_name,
                            "parent_did": parent_did,
                            "mandate": SpawnMandate(
                                parent_did=parent_did
                            ).to_dict(),
                            "config": local.model_dump(mode="json"),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        manager = AgentManager(base_data_dir=tmp_path)

        reconciled = manager._reconcile_spawn_authority_restart_roster(
            MultiAgentConfig(agents={}),
            authority_roots=frozenset({"did:test:surviving-standalone-root"}),
        )

        assert reconciled.agents == {}
        assert SpawnAuthorityRegistry(tmp_path).pending() == ()

    def test_restart_reaps_ownerless_pending_after_empty_sqlite_shell(
        self, tmp_path
    ):
        """Schema creation is not the atomic agent-node birth boundary."""

        child_name = "EmptyShellPendingChild"
        parent_did = "did:test:empty-shell-pending-parent"
        reservation_id = "empty-shell-reservation"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.path.parent.mkdir(parents=True, exist_ok=True)
        registry.path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "records": {},
                    "pending": {
                        reservation_id: {
                            "reservation_id": reservation_id,
                            "child_name": child_name,
                            "parent_did": parent_did,
                            "mandate": SpawnMandate(
                                parent_did=parent_did
                            ).to_dict(),
                            "config": local.model_dump(mode="json"),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        identity_db = local.resolve_data_dir(tmp_path) / "kestrel_prime.db"
        identity_db.parent.mkdir(parents=True)
        with sqlite3.connect(identity_db) as connection:
            connection.execute(
                "CREATE TABLE graph_nodes ("
                "node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, "
                "label TEXT NOT NULL, properties TEXT)"
            )
        manager = AgentManager(base_data_dir=tmp_path)

        reconciled = manager._reconcile_spawn_authority_restart_roster(
            MultiAgentConfig(agents={})
        )

        assert reconciled.agents == {}
        assert SpawnAuthorityRegistry(tmp_path).pending() == ()

    def test_restart_does_not_reap_live_pre_inception_owner(self, tmp_path):
        """A concurrent producer keeps its denial until birth or rollback."""

        child_name = "LivePendingChild"
        parent_did = "did:test:live-pending-parent"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        pending = registry.reserve_pending(
            child_name=child_name,
            parent_did=parent_did,
            mandate=SpawnMandate(parent_did=parent_did),
            config=local,
        )
        manager = AgentManager(base_data_dir=tmp_path)

        manager._reconcile_spawn_authority_restart_roster(
            MultiAgentConfig(agents={})
        )

        assert SpawnAuthorityRegistry(tmp_path).get_pending(
            pending.reservation_id
        ) == pending

    def test_restart_retains_ownerless_pending_after_committed_agent_birth(
        self, tmp_path
    ):
        """A committed agent node keeps the denial across a host crash."""

        child_name = "BornPendingChild"
        child_did = "did:test:born-pending-child"
        parent_did = "did:test:born-pending-parent"
        reservation_id = "born-pending-reservation"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.path.parent.mkdir(parents=True, exist_ok=True)
        registry.path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "records": {},
                    "pending": {
                        reservation_id: {
                            "reservation_id": reservation_id,
                            "child_name": child_name,
                            "parent_did": parent_did,
                            "mandate": SpawnMandate(
                                parent_did=parent_did
                            ).to_dict(),
                            "config": local.model_dump(mode="json"),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        identity_db = local.resolve_data_dir(tmp_path) / "kestrel_prime.db"
        identity_db.parent.mkdir(parents=True)
        with sqlite3.connect(identity_db) as connection:
            connection.execute(
                "CREATE TABLE graph_nodes ("
                "node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, "
                "label TEXT NOT NULL, properties TEXT)"
            )
            connection.execute(
                "INSERT INTO graph_nodes "
                "(node_id, node_type, label, properties) VALUES (?, ?, ?, ?)",
                (child_did, "agent", child_name, "{}"),
            )
        manager = AgentManager(base_data_dir=tmp_path)

        reconciled = manager._reconcile_spawn_authority_restart_roster(
            MultiAgentConfig(agents={child_name: local})
        )

        assert reconciled.agents == {}
        assert (
            SpawnAuthorityRegistry(tmp_path).get_pending(reservation_id) is not None
        )

    def test_restart_retains_ownerless_pending_when_birth_slot_is_unreadable(
        self, monkeypatch, tmp_path
    ):
        """Crash recovery reaps only after positively proving DB absence."""

        child_name = "UnreadableOrphanedPendingChild"
        parent_did = "did:test:unreadable-orphaned-parent"
        reservation_id = "unreadable-orphaned-reservation"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.path.parent.mkdir(parents=True, exist_ok=True)
        registry.path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "records": {},
                    "pending": {
                        reservation_id: {
                            "reservation_id": reservation_id,
                            "child_name": child_name,
                            "parent_did": parent_did,
                            "mandate": SpawnMandate(parent_did=parent_did).to_dict(),
                            "config": local.model_dump(mode="json"),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        identity_db = local.resolve_data_dir(tmp_path) / "kestrel_prime.db"
        identity_db.parent.mkdir(parents=True)
        identity_db.touch()
        original_stat = Path.stat
        original_isfile = os.path.isfile

        def unreadable_identity(path, *args, **kwargs):
            if path == identity_db:
                raise PermissionError("identity slot is temporarily unreadable")
            return original_stat(path, *args, **kwargs)

        def suppressed_unreadable_isfile(path):
            if Path(path) == identity_db:
                return False
            return original_isfile(path)

        monkeypatch.setattr(Path, "stat", unreadable_identity)
        monkeypatch.setattr(os.path, "isfile", suppressed_unreadable_isfile)
        manager = AgentManager(base_data_dir=tmp_path)

        reconciled = manager._reconcile_spawn_authority_restart_roster(
            MultiAgentConfig(agents={child_name: local})
        )

        assert reconciled.agents == {}
        assert SpawnAuthorityRegistry(tmp_path).get_pending(reservation_id) is not None

    def test_restart_retires_expired_cold_witness_and_descendant(self, tmp_path):
        """Cold desired state cannot preserve an elapsed signed lifetime."""

        parent_name = "ExpiredColdParent"
        parent_did = "did:test:expired-cold-parent"
        child_name = "ColdDescendant"
        child_did = "did:test:cold-descendant"
        expired_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        parent_config = LocalAgentConfig(
            data_dir=Path("agent_data") / parent_name,
            port=8802,
            autostart=False,
        )
        child_config = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8803,
            autostart=False,
        )
        for config in (parent_config, child_config):
            data_dir = config.resolve_data_dir(tmp_path)
            data_dir.mkdir(parents=True)
            (data_dir / "kestrel_prime.db").touch()
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=parent_name,
            child_did=parent_did,
            mandate=SpawnMandate(
                parent_did="did:test:cold-root",
                child_did=parent_did,
                ttl_seconds=1,
                created_at=expired_at,
                parent_signature="signed-expired-parent",
            ),
            config=parent_config,
        )
        registry.record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=SpawnMandate(
                parent_did=parent_did,
                child_did=child_did,
                ttl_seconds=3600,
                parent_signature="signed-live-descendant",
            ),
            config=child_config,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        roster = MultiAgentConfig(
            agents={parent_name: parent_config, child_name: child_config}
        )

        with patch(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did_sync",
            side_effect=lambda path, **_kwargs: (
                parent_did if Path(path).name == parent_name else child_did
            ),
        ):
            reconciled = manager._reconcile_spawn_authority_restart_roster(roster)

        assert reconciled.agents == {}
        assert registry.get(parent_did).retired
        assert registry.get(child_did).retired
        assert manager._spawn_cap_slots_in_use() == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("autostart", [False, True])
    async def test_restart_arms_expiry_for_valid_witness_left_cold(
        self, monkeypatch, tmp_path, autostart
    ):
        """A skipped or failed cold load still releases its finite host slot."""

        child_name = "ValidColdExpiryChild"
        child_did = "did:test:valid-cold-expiry-child"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
            autostart=autostart,
        )
        mandate = SpawnMandate(
            parent_did="did:test:valid-cold-expiry-parent",
            child_did=child_did,
            ttl_seconds=1,
            parent_signature="signed-valid-cold-expiry",
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=local,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        monkeypatch.setattr(
            SpawnedAgentLifecycle,
            "_remaining_ttl_seconds",
            staticmethod(lambda _created_at, _ttl_seconds: 0.01),
        )
        if autostart:
            manager._initialize_agent = AsyncMock(
                side_effect=RuntimeError("cold child cannot load")
            )

        loaded = await manager.load_from_config(
            MultiAgentConfig(agents={child_name: local})
        )

        assert loaded == 0
        await asyncio.sleep(0.05)
        assert registry.get(child_did).retired
        assert manager._spawn_cap_slots_in_use() == 0

    def test_cold_expiry_does_not_tombstone_replacement_identity(self, tmp_path):
        """An elapsed old witness cannot retire a new birth in the same slot."""

        child_name = "ReplacedExpiredColdChild"
        expired_did = "did:test:expired-cold-original"
        replacement_did = "did:test:expired-cold-replacement"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
            autostart=False,
        )
        data_dir = local.resolve_data_dir(tmp_path)
        data_dir.mkdir(parents=True)
        (data_dir / "kestrel_prime.db").touch()
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=child_name,
            child_did=expired_did,
            mandate=SpawnMandate(
                parent_did="did:test:expired-cold-root",
                child_did=expired_did,
                ttl_seconds=1,
                created_at=(
                    datetime.now(timezone.utc) - timedelta(seconds=10)
                ).isoformat(),
                parent_signature="signed-expired-original",
            ),
            config=local,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        roster = MultiAgentConfig(agents={child_name: local})

        with patch(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did_sync",
            return_value=replacement_did,
        ):
            reconciled = manager._reconcile_spawn_authority_restart_roster(roster)

        assert reconciled.agents == roster.agents
        assert registry.get(expired_did).retired
        assert not (data_dir / ".kestrel-spawn-retired").exists()

    def test_pending_spawn_authority_promotes_atomically_to_signed_witness(
        self, tmp_path
    ):
        """Promotion leaves exactly one active authority state for the slot."""

        parent_did = "did:test:pending-promotion-parent"
        child_did = "did:test:pending-promotion-child"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / "PromotedChild",
            port=8802,
        )
        proposal = SpawnMandate(parent_did=parent_did, purpose="atomic promotion")
        registry = SpawnAuthorityRegistry(tmp_path)
        pending = registry.reserve_pending(
            child_name="PromotedChild",
            parent_did=parent_did,
            mandate=proposal,
            config=local,
        )
        private_key, _ = generate_secp256k1_keypair()
        final = replace(proposal, child_did=child_did)
        sign_mandate(final, private_key)

        promoted = registry.promote_pending(
            reservation_id=pending.reservation_id,
            child_name="PromotedChild",
            child_did=child_did,
            mandate=final,
            config=local,
            proposal_created_at=proposal.created_at,
        )

        reloaded = SpawnAuthorityRegistry(tmp_path)
        assert promoted.active
        assert reloaded.pending() == ()
        assert reloaded.get(child_did) == promoted

    def test_registry_v1_is_read_and_upgraded_before_pending_reservation(
        self, tmp_path
    ):
        """Existing hosts migrate safely; older binaries reject later v2 writes."""

        registry = SpawnAuthorityRegistry(tmp_path)
        child_did = "did:test:v1-child"
        registry.record_active(
            child_name="V1Child",
            child_did=child_did,
            mandate=SpawnMandate(
                parent_did="did:test:v1-parent",
                child_did=child_did,
                parent_signature="v1-signature",
            ),
            config=LocalAgentConfig(
                data_dir=Path("agent_data") / "V1Child",
                port=8801,
            ),
        )
        payload = json.loads(registry.path.read_text(encoding="utf-8"))
        payload["version"] = 1
        payload.pop("pending")
        registry.path.write_text(json.dumps(payload), encoding="utf-8")

        assert registry.get(child_did) is not None
        registry.reserve_pending(
            child_name="V2PendingChild",
            parent_did="did:test:v2-parent",
            mandate=SpawnMandate(parent_did="did:test:v2-parent"),
            config=LocalAgentConfig(
                data_dir=Path("agent_data") / "V2PendingChild",
                port=8802,
            ),
        )

        upgraded = json.loads(registry.path.read_text(encoding="utf-8"))
        assert upgraded["version"] == 2
        assert registry.get(child_did) is not None
        assert len(registry.pending()) == 1

    def test_standalone_restart_selects_only_own_transitive_authority_tree(
        self,
        tmp_path,
    ):
        root_a = "did:test:standalone-root-a"
        root_b = "did:test:standalone-root-b"
        child_a = "did:test:standalone-child-a"
        grandchild_a = "did:test:standalone-grandchild-a"
        child_b = "did:test:standalone-child-b"
        registry = SpawnAuthorityRegistry(tmp_path)

        def record(name, parent_did, child_did, port):
            registry.record_active(
                child_name=name,
                child_did=child_did,
                mandate=SpawnMandate(
                    parent_did=parent_did,
                    child_did=child_did,
                    parent_signature=f"signature:{name}",
                ),
                config=LocalAgentConfig(
                    data_dir=Path("agent_data") / name,
                    port=port,
                ),
            )

        record("ChildA", root_a, child_a, 8801)
        record("GrandchildA", child_a, grandchild_a, 8802)
        # A sibling standalone root can reuse the same in-process port number;
        # its process and authority tree are distinct and must be filtered out
        # before roster validation.
        record("ChildB", root_b, child_b, 8801)
        shared_roster = MultiAgentConfig(
            agents={
                "RootA": LocalAgentConfig(
                    data_dir=Path("agent_data") / "RootA",
                    port=8810,
                ),
                "RootB": LocalAgentConfig(
                    data_dir=Path("agent_data") / "RootB",
                    port=8811,
                ),
            }
        )
        config_path = tmp_path / "multi_agent.toml"
        shared_roster.save(config_path)
        manager = AgentManager(
            base_data_dir=tmp_path,
            startup_roster_enabled=False,
        )

        reconciled = manager._reconcile_spawn_authority_restart_roster(
            MultiAgentConfig(agents={}),
            authority_roots=frozenset({root_a}),
        )

        assert set(reconciled.agents) == {"ChildA", "GrandchildA"}
        assert MultiAgentConfig.from_file(config_path) == shared_roster

    def test_scoped_restart_retires_expired_witness_from_sibling_root(
        self,
        tmp_path,
    ):
        """A host-global expired witness cannot consume a sibling root's cap."""

        scoped_root = "did:test:scoped-healthy-root"
        offline_root = "did:test:offline-expired-root"
        expired_child_did = "did:test:offline-expired-child"
        expired_name = "OfflineExpiredChild"
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=expired_name,
            child_did=expired_child_did,
            mandate=SpawnMandate(
                parent_did=offline_root,
                child_did=expired_child_did,
                ttl_seconds=1,
                created_at=(
                    datetime.now(timezone.utc) - timedelta(seconds=10)
                ).isoformat(),
                parent_signature="signed-offline-expired-child",
            ),
            config=LocalAgentConfig(
                data_dir=Path("agent_data") / expired_name,
                port=8801,
                autostart=False,
            ),
        )
        manager = AgentManager(
            base_data_dir=tmp_path,
            startup_roster_enabled=False,
        )
        manager._max_spawned_agents = 1
        assert manager._spawn_cap_slots_in_use() == 1

        reconciled = manager._reconcile_spawn_authority_restart_roster(
            MultiAgentConfig(agents={}),
            authority_roots=frozenset({scoped_root}),
        )

        assert reconciled.agents == {}
        assert registry.get(expired_child_did).retired
        assert manager._spawn_cap_slots_in_use() == 0

    def test_standalone_root_does_not_claim_sibling_child_name(self, tmp_path):
        """A root name is outside sibling standalone spawn-slot authority."""

        root_did = "did:test:standalone-own-root"
        sibling_child_did = "did:test:sibling-same-name-child"
        sibling_parent_did = "did:test:sibling-root"
        shared_name = "SharedStandaloneName"
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=shared_name,
            child_did=sibling_child_did,
            mandate=SpawnMandate(
                parent_did=sibling_parent_did,
                child_did=sibling_child_did,
                parent_signature="signed-sibling-child",
            ),
            config=LocalAgentConfig(
                data_dir=Path("agent_data") / "SiblingChild",
                port=8802,
            ),
        )
        root = _make_mock_agent(root_did)
        manager = AgentManager(
            base_data_dir=tmp_path,
            startup_roster_enabled=False,
        )

        manager._verify_agent_authority(shared_name, root)

    def test_registry_mutation_lock_serializes_independent_managers(self, tmp_path):
        first = SpawnAuthorityRegistry(tmp_path)
        second = SpawnAuthorityRegistry(tmp_path)
        first_entered = threading.Event()
        second_entered = threading.Event()
        release_first = threading.Event()
        failures: list[BaseException] = []
        first_read = first._read_state
        second_read = second._read_state

        def delayed_first_read():
            records = first_read()
            first_entered.set()
            if not release_first.wait(timeout=2):
                raise RuntimeError("registry lock test timed out")
            return records

        def observed_second_read():
            second_entered.set()
            return second_read()

        first._read_state = delayed_first_read
        second._read_state = observed_second_read

        def record(registry, name, child_did, port):
            try:
                registry.record_active(
                    child_name=name,
                    child_did=child_did,
                    mandate=SpawnMandate(
                        parent_did=f"did:test:parent:{name}",
                        child_did=child_did,
                        parent_signature=f"signature:{name}",
                    ),
                    config=LocalAgentConfig(
                        data_dir=Path("agent_data") / name,
                        port=port,
                    ),
                )
            except BaseException as error:
                failures.append(error)

        first_thread = threading.Thread(
            target=record,
            args=(first, "LockedA", "did:test:locked-a", 8801),
        )
        second_thread = threading.Thread(
            target=record,
            args=(second, "LockedB", "did:test:locked-b", 8802),
        )
        first_thread.start()
        assert first_entered.wait(timeout=1)
        second_thread.start()
        try:
            assert not second_entered.wait(timeout=0.1)
        finally:
            release_first.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert failures == []
        assert {record.child_did for record in first.records()} == {
            "did:test:locked-a",
            "did:test:locked-b",
        }

    def test_registry_cap_is_enforced_inside_shared_mutation_lock(self, tmp_path):
        """Independent standalone producers cannot both take the final slot."""

        first = SpawnAuthorityRegistry(tmp_path)
        second = SpawnAuthorityRegistry(tmp_path)
        errors: list[BaseException] = []
        accepted: list[str] = []
        start = threading.Barrier(2)

        def reserve(registry, name, port):
            try:
                start.wait(timeout=1)
                registry.reserve_pending(
                    child_name=name,
                    parent_did=f"did:test:parent:{name}",
                    mandate=SpawnMandate(parent_did=f"did:test:parent:{name}"),
                    config=LocalAgentConfig(
                        data_dir=Path("agent_data") / name,
                        port=port,
                    ),
                    max_authority_slots=1,
                )
                accepted.append(name)
            except BaseException as error:
                errors.append(error)

        first_thread = threading.Thread(
            target=reserve,
            args=(first, "FinalSlotA", 8801),
        )
        second_thread = threading.Thread(
            target=reserve,
            args=(second, "FinalSlotB", 8802),
        )
        first_thread.start()
        second_thread.start()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert len(accepted) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        assert "spawned-agent cap" in str(errors[0])
        assert len(SpawnAuthorityRegistry(tmp_path).pending()) == 1

    def test_registry_backfill_cap_is_atomic_with_cold_authority(self, tmp_path):
        """A stale precheck cannot overfill the durable authority registry."""

        registry = SpawnAuthorityRegistry(tmp_path)
        first_did = "did:test:atomic-backfill-first"
        registry.record_active(
            child_name="AtomicBackfillFirst",
            child_did=first_did,
            mandate=SpawnMandate(
                parent_did="did:test:atomic-backfill-parent",
                child_did=first_did,
                parent_signature="signed-atomic-backfill-first",
            ),
            config=LocalAgentConfig(
                data_dir=Path("agent_data") / "AtomicBackfillFirst",
                port=8801,
            ),
        )
        second_did = "did:test:atomic-backfill-second"

        with pytest.raises(ValueError, match="spawned-agent cap"):
            registry.record_active(
                child_name="AtomicBackfillSecond",
                child_did=second_did,
                mandate=SpawnMandate(
                    parent_did="did:test:atomic-backfill-parent",
                    child_did=second_did,
                    parent_signature="signed-atomic-backfill-second",
                ),
                config=LocalAgentConfig(
                    data_dir=Path("agent_data") / "AtomicBackfillSecond",
                    port=8802,
                ),
                max_authority_slots=1,
            )

        assert {witness.child_did for witness in registry.records()} == {first_did}

    @pytest.mark.asyncio
    async def test_standalone_retry_skips_exact_child_already_loaded(self, tmp_path):
        parent_did = "did:test:standalone-partial-parent"
        child_a_did = "did:test:standalone-loaded-child"
        child_b_did = "did:test:standalone-retry-child"
        parent = _make_mock_agent(parent_did)
        parent._private_key, _ = generate_secp256k1_keypair()
        parent.identity = None
        parent._persisted_spawn_mandate = None
        config_a = LocalAgentConfig(
            data_dir=Path("agent_data") / "LoadedChild",
            port=8801,
        )
        config_b = LocalAgentConfig(
            data_dir=Path("agent_data") / "RetryChild",
            port=8802,
        )
        mandate_a = sign_mandate(
            SpawnMandate(parent_did=parent_did, child_did=child_a_did),
            parent._private_key,
        )
        mandate_b = sign_mandate(
            SpawnMandate(parent_did=parent_did, child_did=child_b_did),
            parent._private_key,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name="LoadedChild",
            child_did=child_a_did,
            mandate=mandate_a,
            config=config_a,
        )
        registry.record_active(
            child_name="RetryChild",
            child_did=child_b_did,
            mandate=mandate_b,
            config=config_b,
        )
        manager = AgentManager(
            base_data_dir=tmp_path,
            startup_roster_enabled=False,
        )
        manager._register_agent("StandaloneParent", parent)
        child_a = _make_mock_agent(child_a_did)
        child_a._persisted_spawn_mandate = mandate_a
        manager._created_configs["LoadedChild"] = config_a
        manager._register_agent("LoadedChild", child_a)
        child_b = _make_mock_agent(child_b_did)
        child_b._persisted_spawn_mandate = mandate_b
        manager._initialize_agent = AsyncMock(return_value=child_b)
        manager._on_agent_registered = AsyncMock()
        manager._run_hosted_agent_ready_hooks = AsyncMock()

        loaded = await manager.load_from_config(
            MultiAgentConfig(agents={}),
            authority_roots=frozenset({parent_did}),
        )

        assert loaded == 1
        manager._initialize_agent.assert_awaited_once_with("RetryChild", config_b)
        assert manager.get_agent("LoadedChild") is child_a
        assert manager.get_agent("RetryChild") is child_b
        assert manager.init_failures == []

    @pytest.mark.asyncio
    async def test_standalone_shutdown_preserves_cold_stop_and_owns_finite_expiry(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Stop retains persistent restart authority but never abandons a TTL."""

        root_did = "did:test:cold-shutdown-root"
        child_did = "did:test:cold-shutdown-child"
        grandchild_did = "did:test:cold-shutdown-grandchild"
        registry = SpawnAuthorityRegistry(tmp_path)
        for name, did, parent_did, port, ttl_seconds in (
            ("ColdShutdownChild", child_did, root_did, 8802, 0),
            (
                "ColdShutdownGrandchild",
                grandchild_did,
                child_did,
                8803,
                3600,
            ),
        ):
            registry.record_active(
                child_name=name,
                child_did=did,
                mandate=SpawnMandate(
                    parent_did=parent_did,
                    child_did=did,
                    ttl_seconds=ttl_seconds,
                    parent_signature=f"signed-{name}",
                ),
                config=LocalAgentConfig(
                    data_dir=Path("agent_data") / name,
                    port=port,
                    autostart=False,
                ),
            )
        manager = AgentManager(
            base_data_dir=tmp_path,
            startup_roster_enabled=False,
        )
        root = SimpleNamespace(did=root_did, agent_id=root_did)
        monkeypatch.setattr(
            SpawnedAgentLifecycle,
            "_remaining_ttl_seconds",
            staticmethod(lambda *_args: 0.0),
        )

        await manager.shutdown_spawn_authority_tree(root)
        for _ in range(20):
            grandchild = registry.get(grandchild_did)
            lifecycle = getattr(manager, "_lifecycle", None)
            if (
                grandchild is not None
                and grandchild.retired
                and isinstance(lifecycle, SpawnedAgentLifecycle)
                and not lifecycle._cold_ttl_tasks
            ):
                break
            await asyncio.sleep(0)

        # A normal Stop is explicitly restartable; only Hold/terminal expiry
        # retires its durable authority.  The finite cold descendant did still
        # receive an expiry owner and reached terminal retirement.
        assert registry.get(child_did).active
        assert registry.get(grandchild_did).retired

    @pytest.mark.asyncio
    async def test_standalone_shutdown_normalizes_rotated_parent_ttl_handoff(
        self,
        tmp_path,
    ):
        """A successor-signed child keeps one stable-parent deadline after Stop."""

        stable_parent_did = "did:pkh:eip155:1:0xStandaloneRotatedParent"
        signing_parent_did = "did:web:example.test:standalone-rotated-parent"
        child_name = "StandaloneRotatedChild"
        child_did = "did:test:standalone-rotated-child"
        mandate = SpawnMandate(
            parent_did=signing_parent_did,
            child_did=child_did,
            ttl_seconds=3600,
            parent_signature="signed-standalone-rotated-child",
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
        manager = AgentManager(
            base_data_dir=tmp_path,
            startup_roster_enabled=False,
        )
        child = _make_mock_agent(child_did)
        manager._agents[child_name] = child
        manager._agent_names[child_did] = child_name
        manager._parent_children[stable_parent_did] = [child_name]
        manager._child_mandates[child_name] = mandate
        manager._created_configs[child_name] = config
        lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle = lifecycle
        root = SimpleNamespace(
            did=stable_parent_did,
            agent_id=stable_parent_did,
            identity=SimpleNamespace(
                legacy_did=stable_parent_did,
                new_did=signing_parent_did,
            ),
        )

        await manager.shutdown_spawn_authority_tree(root)

        key = (child_name.casefold(), child_did)
        owner = lifecycle._cold_ttl_tasks.get(key)
        try:
            assert manager.get_agent(child_name) is None
            assert registry.get(child_did).active
            assert owner is not None and not owner.done()
        finally:
            if owner is not None:
                owner.cancel()

    @pytest.mark.asyncio
    async def test_generic_destructive_removal_retires_spawn_witness(self, tmp_path):
        child_name = "DirectDeleteChild"
        child_did = "did:test:direct-delete-child"
        parent_did = "did:test:direct-delete-parent"
        config = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        mandate = SpawnMandate(
            parent_did=parent_did,
            child_did=child_did,
            parent_signature="direct-delete-signature",
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=config,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        child = _make_mock_agent(child_did)
        child._persisted_spawn_mandate = mandate
        manager._agents[child_name] = child
        manager._agent_names[child_did] = child_name
        manager._parent_children[parent_did] = [child_name]
        manager._child_mandates[child_name] = mandate
        manager._created_configs[child_name] = config

        with pytest.raises(RuntimeOffboardingNotPerformedError):
            await manager.remove_agent(child_name, offboard_runtime=True)

        witness = registry.get(child_did)
        assert witness is not None and witness.retired
        assert manager.get_agent(child_name) is None
        assert manager.reconcile_spawn_authority_restart_roster(
            MultiAgentConfig(agents={})
        ).agents == {}

    @pytest.mark.asyncio
    async def test_destructive_parent_removal_refuses_nonbudgeted_descendants(
        self,
        tmp_path,
    ):
        """Deleting a parent cannot strand a still-active child witness."""

        parent_name = "DeleteParent"
        parent_did = "did:test:delete-parent"
        child_name = "NonBudgetedChild"
        child_did = "did:test:nonbudgeted-child"
        manager = AgentManager(base_data_dir=tmp_path)
        parent = _make_mock_agent(parent_did)
        child = _make_mock_agent(child_did)
        manager._agents.update({parent_name: parent, child_name: child})
        manager._agent_names.update({parent_did: parent_name, child_did: child_name})
        manager._parent_children[parent_did] = [child_name]
        manager._child_mandates[child_name] = SpawnMandate(
            parent_did=parent_did,
            child_did=child_did,
            parent_signature="nonbudgeted-child-authority",
        )

        with pytest.raises(RuntimeError, match="active spawned descendants"):
            await manager.remove_agent(parent_name, offboard_runtime=True)

        assert manager.get_agent(parent_name) is parent
        assert manager.get_agent(child_name) is child

    @pytest.mark.asyncio
    async def test_terminal_descendant_spawn_fence_joins_and_refuses_late_spawn(
        self,
        tmp_path,
    ):
        """A terminal snapshot owns every earlier spawn and excludes later ones."""

        manager = AgentManager(base_data_dir=tmp_path)
        parent = _make_mock_agent("did:test:terminal-spawn-parent")
        parent.features = {}
        manager._agents["TerminalSpawnParent"] = parent
        manager._agent_names[parent.agent_id] = "TerminalSpawnParent"
        release = asyncio.Event()

        async def admitted_spawn_owner():
            await release.wait()

        owner = asyncio.create_task(admitted_spawn_owner())
        admission = AgentOperationAdmission(
            name="EarlierChild",
            canonical_name="earlierchild",
            kind="spawn",
            registration_epoch=manager._agent_registration_shutdown_epoch,
            owner_task=owner,
            spawn_task=owner,
            spawn_parent=parent,
        )
        manager._agent_operations[admission.canonical_name] = admission

        fence = asyncio.create_task(
            manager.begin_terminal_descendant_spawn_fence(parent.agent_id)
        )
        await asyncio.sleep(0)
        assert not fence.done()
        release.set()
        token = await fence

        with pytest.raises(RuntimeError, match="terminal retirement"):
            await manager.spawn_agent(
                "LateChild",
                parent,
                SpawnMandate(parent_did=parent.agent_id),
            )

        manager.end_terminal_descendant_spawn_fence(token)
        manager._agent_operations.pop(admission.canonical_name, None)

    @pytest.mark.asyncio
    async def test_terminal_descendant_spawn_fence_covers_existing_subtree(
        self,
        tmp_path,
    ):
        """A grandchild spawn cannot slip outside the parent's tree snapshot."""

        manager = AgentManager(base_data_dir=tmp_path)
        root = _make_mock_agent("did:test:terminal-subtree-root")
        child = _make_mock_agent("did:test:terminal-subtree-child")
        child.features = {}
        manager._agents.update({"FenceRoot": root, "FenceChild": child})
        manager._agent_names.update(
            {root.agent_id: "FenceRoot", child.agent_id: "FenceChild"}
        )
        manager._parent_children[root.agent_id] = ["FenceChild"]
        manager._child_mandates["FenceChild"] = SpawnMandate(
            parent_did=root.agent_id,
            child_did=child.agent_id,
            parent_signature="signed-subtree-child",
        )
        release = asyncio.Event()

        async def admitted_grandchild_spawn():
            await release.wait()

        owner = asyncio.create_task(admitted_grandchild_spawn())
        admission = AgentOperationAdmission(
            name="EarlierGrandchild",
            canonical_name="earliergrandchild",
            kind="spawn",
            registration_epoch=manager._agent_registration_shutdown_epoch,
            owner_task=owner,
            spawn_task=owner,
            spawn_parent=child,
        )
        manager._agent_operations[admission.canonical_name] = admission

        fence = asyncio.create_task(
            manager.begin_terminal_descendant_spawn_fence(root.agent_id)
        )
        await asyncio.sleep(0)
        assert not fence.done()
        release.set()
        token = await fence

        with pytest.raises(RuntimeError, match="terminal retirement"):
            await manager.spawn_agent(
                "LateGrandchild",
                child,
                SpawnMandate(parent_did=child.agent_id),
            )

        manager.end_terminal_descendant_spawn_fence(token)
        manager._agent_operations.pop(admission.canonical_name, None)

    @pytest.mark.asyncio
    async def test_terminal_descendant_spawn_fence_covers_cold_durable_subtree(
        self,
        tmp_path,
    ):
        """Cold descendants are fenced before a terminal tree is snapshotted."""

        root_did = "did:test:cold-fence-root"
        child_did = "did:test:cold-fence-child"
        grandchild_did = "did:test:cold-fence-grandchild"
        registry = SpawnAuthorityRegistry(tmp_path)
        for name, did, parent_did, port in (
            ("ColdFenceChild", child_did, root_did, 8802),
            ("ColdFenceGrandchild", grandchild_did, child_did, 8803),
        ):
            registry.record_active(
                child_name=name,
                child_did=did,
                mandate=SpawnMandate(
                    parent_did=parent_did,
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
        token = await manager.begin_terminal_descendant_spawn_fence(root_did)
        try:
            assert manager._terminal_descendant_spawn_fence_members[token.nonce] == {
                root_did,
                child_did,
                grandchild_did,
            }
        finally:
            manager.end_terminal_descendant_spawn_fence(token)

    def test_restartable_stopped_child_still_consumes_spawn_capacity(self, tmp_path):
        """A retained child cannot free its fleet slot merely by stopping."""

        child_name = "ColdCapacityChild"
        child_did = "did:test:cold-capacity-child"
        mandate = SpawnMandate(
            parent_did="did:test:cold-capacity-parent",
            child_did=child_did,
            ttl_seconds=0,
            parent_signature="signed-cold-capacity",
        )
        SpawnAuthorityRegistry(tmp_path).record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=LocalAgentConfig(
                data_dir=Path("agent_data") / child_name,
                port=8802,
            ),
        )
        manager = AgentManager(base_data_dir=tmp_path)
        manager._max_spawned_agents = 1

        assert manager._child_mandates == {}
        assert manager._spawn_cap_slots_in_use() == 1

    def test_cold_preflight_reservation_deduplicates_durable_capacity(self, tmp_path):
        """One cold child consumes one slot while moving into its projection."""

        child_name = "PreflightCapacityChild"
        child_did = "did:test:preflight-capacity-child"
        mandate = SpawnMandate(
            parent_did="did:test:preflight-capacity-parent",
            child_did=child_did,
            ttl_seconds=0,
            parent_signature="signed-preflight-capacity",
        )
        SpawnAuthorityRegistry(tmp_path).record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=LocalAgentConfig(
                data_dir=Path("agent_data") / child_name,
                port=8802,
            ),
        )
        manager = AgentManager(base_data_dir=tmp_path)
        child = _make_mock_agent(child_did)
        manager._preflight_spawn_reservations[
            manager._canonical_agent_name(child_name)
        ] = (child, child_did)

        assert manager._child_mandates == {}
        assert manager._spawn_cap_slots_in_use() == 1

    @pytest.mark.asyncio
    async def test_parent_offboard_refuses_restartable_stopped_descendant(
        self,
        tmp_path,
    ):
        """Cold durable descendants remain a destructive-offboard dependency."""

        parent_name = "ColdDescendantParent"
        parent_did = "did:test:cold-descendant-parent"
        child_name = "ColdDescendant"
        child_did = "did:test:cold-descendant"
        mandate = SpawnMandate(
            parent_did=parent_did,
            child_did=child_did,
            ttl_seconds=0,
            parent_signature="signed-cold-descendant",
        )
        SpawnAuthorityRegistry(tmp_path).record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=LocalAgentConfig(
                data_dir=Path("agent_data") / child_name,
                port=8802,
            ),
        )
        manager = AgentManager(base_data_dir=tmp_path)
        parent = _make_mock_agent(parent_did)
        manager._agents[parent_name] = parent
        manager._agent_names[parent_did] = parent_name
        manager._remove_agent_serialized = AsyncMock(return_value=True)

        with pytest.raises(RuntimeError, match="active spawned descendants"):
            await manager.remove_agent(parent_name, offboard_runtime=True)

        manager._remove_agent_serialized.assert_not_awaited()
        assert manager.get_agent(parent_name) is parent

    @pytest.mark.asyncio
    async def test_destructive_child_retirement_precedes_startup_row_removal(
        self,
        tmp_path,
    ):
        """A crash after roster removal must find a durable restart denial."""

        parent_did = "did:test:ordered-offboard-parent"
        child_name = "OrderedOffboardChild"
        child_did = "did:test:ordered-offboard-child"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        config_path = tmp_path / "multi_agent.toml"
        MultiAgentConfig(agents={child_name: local}).save(config_path)
        mandate = SpawnMandate(
            parent_did=parent_did,
            child_did=child_did,
            ttl_seconds=0,
            parent_signature="signed-ordered-offboard",
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=local,
        )
        manager = AgentManager(
            base_data_dir=tmp_path,
            startup_config_path=config_path,
        )
        child = _make_mock_agent(child_did)
        manager._agents[child_name] = child
        manager._agent_names[child_did] = child_name
        manager._parent_children[parent_did] = [child_name]
        manager._child_mandates[child_name] = mandate
        observed_states = []

        def crash_after_observing_retirement(*_args, **_kwargs):
            observed_states.append(registry.get(child_did).state)
            raise SystemExit("simulated crash after startup-row removal")

        with patch.object(
            manager,
            "_withdraw_committed_spawn_startup_registration",
            side_effect=crash_after_observing_retirement,
        ), pytest.raises(SystemExit, match="simulated crash"):
            await manager.terminate_child(
                parent_did,
                child_name,
                offboard_runtime=True,
            )

        assert observed_states == ["retiring"]
        assert registry.get(child_did).state == "retiring"

    @pytest.mark.asyncio
    async def test_refused_destructive_child_removal_reopens_restart_authority(
        self,
        tmp_path,
    ):
        """A live child must not remain retiring after ordinary refusal."""

        parent_did = "did:test:refused-offboard-parent"
        child_name = "RefusedOffboardChild"
        child_did = "did:test:refused-offboard-child"
        mandate = SpawnMandate(
            parent_did=parent_did,
            child_did=child_did,
            ttl_seconds=0,
            parent_signature="signed-refused-offboard",
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
        child = _make_mock_agent(child_did)
        manager._agents[child_name] = child
        manager._agent_names[child_did] = child_name
        manager._parent_children[parent_did] = [child_name]
        manager._child_mandates[child_name] = mandate
        manager.remove_agent = AsyncMock(side_effect=RuntimeError("refused"))

        with pytest.raises(RuntimeError, match="refused"):
            await manager.terminate_child(
                parent_did,
                child_name,
                offboard_runtime=True,
            )

        assert manager.get_agent(child_name) is child
        assert registry.get(child_did).active

    @pytest.mark.asyncio
    async def test_scheduler_cold_wake_loads_cold_authority_parent_first(
        self,
        tmp_path,
    ):
        """A due child schedule restores its cold governing chain on demand."""

        parent_name = "SchedulerColdParent"
        child_name = "SchedulerColdChild"
        parent_did = "did:test:scheduler-cold-parent"
        child_did = "did:test:scheduler-cold-child"
        parent, mandate = _signed_restored_mandate(parent_did, child_did)
        child = _make_mock_agent(child_did)
        child._persisted_spawn_mandate = mandate
        parent_config = LocalAgentConfig(
            data_dir=Path("agent_data") / parent_name,
            port=8801,
            autostart=False,
        )
        child_config = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
            autostart=False,
        )
        SpawnAuthorityRegistry(tmp_path).record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=child_config,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        manager._seed_scheduler_authority(
            {
                parent_did: (parent_name, parent_config),
                child_did: (child_name, child_config),
            }
        )
        load_order = []

        async def initialize(name, _config, **_kwargs):
            load_order.append(name)
            return parent if name == parent_name else child

        manager._initialize_agent = AsyncMock(side_effect=initialize)
        manager._on_agent_registered = AsyncMock()
        manager._run_hosted_agent_ready_hooks = AsyncMock()

        loaded = await manager.load_agent(
            child_name,
            child_config,
            expected_agent_id=child_did,
        )

        assert loaded is child
        assert load_order == [parent_name, child_name]
        assert manager.get_agent(parent_name) is parent
        assert manager.get_agent(child_name) is child

    @pytest.mark.asyncio
    @pytest.mark.parametrize("preexisting_retirement", [False, True])
    async def test_refused_destructive_removal_reopens_owned_spawn_witness(
        self,
        tmp_path,
        preexisting_retirement,
    ):
        child_name = "RefusedDeleteChild"
        child_did = "did:test:refused-delete-child"
        parent_did = "did:test:refused-delete-parent"
        config = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        mandate = SpawnMandate(
            parent_did=parent_did,
            child_did=child_did,
            parent_signature="refused-delete-signature",
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=config,
        )
        if preexisting_retirement:
            registry.begin_retirement(
                child_name=child_name,
                child_did=child_did,
            )
        manager = AgentManager(base_data_dir=tmp_path)
        child = _make_mock_agent(child_did)
        child.shutdown = AsyncMock(side_effect=RuntimeError("shutdown refused"))
        child._persisted_spawn_mandate = mandate
        manager._agents[child_name] = child
        manager._agent_names[child_did] = child_name
        manager._parent_children[parent_did] = [child_name]
        manager._child_mandates[child_name] = mandate
        manager._created_configs[child_name] = config

        assert await manager.remove_agent(child_name, offboard_runtime=True) is False

        witness = registry.get(child_did)
        assert witness is not None
        assert witness.state == (
            "retiring" if preexisting_retirement else "active"
        )
        assert manager.get_agent(child_name) is child

    def test_host_spawn_witness_preserves_operator_config_edits(self, tmp_path):
        config_path = tmp_path / "multi_agent.toml"
        recovery_snapshot = LocalAgentConfig(
            data_dir=Path("agent_data") / "EditedChild",
            port=8801,
        )
        operator_config = recovery_snapshot.model_copy(
            update={"port": 8802, "autostart": False}
        )
        MultiAgentConfig(agents={"EditedChild": operator_config}).save(config_path)
        private_key, _ = generate_secp256k1_keypair()
        mandate = sign_mandate(
            SpawnMandate(
                parent_did="did:test:edited-parent",
                child_did="did:test:edited-child",
            ),
            private_key,
        )
        SpawnAuthorityRegistry(tmp_path).record_active(
            child_name="EditedChild",
            child_did=mandate.child_did,
            mandate=mandate,
            config=recovery_snapshot,
        )
        manager = AgentManager(
            base_data_dir=tmp_path,
            startup_config_path=config_path,
        )

        reconciled = manager._reconcile_spawn_authority_restart_roster(
            MultiAgentConfig.from_file(config_path)
        )

        assert reconciled.agents["EditedChild"] == operator_config
        assert MultiAgentConfig.from_file(config_path).agents["EditedChild"] == operator_config

    def test_terminal_retirement_tombstones_host_spawn_witness(self, tmp_path):
        child_name = "FinishedChild"
        child_did = "did:test:finished-child"
        private_key, _ = generate_secp256k1_keypair()
        mandate = sign_mandate(
            SpawnMandate(
                parent_did="did:test:finished-parent",
                child_did=child_did,
            ),
            private_key,
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

        assert (
            manager.record_expired_spawn_retirement(
                child_name,
                expected_child_did=child_did,
            )
            is None
        )
        witness = registry.get(child_did)
        assert witness is not None
        assert witness.retired is True

    def test_terminal_retirement_intent_denies_restart_until_live_refusal_cancels(
        self,
        tmp_path,
    ):
        child_name = "CrashOrderedChild"
        child_did = "did:test:crash-ordered-child"
        parent_did = "did:test:crash-ordered-parent"
        private_key, _ = generate_secp256k1_keypair()
        mandate = sign_mandate(
            SpawnMandate(parent_did=parent_did, child_did=child_did),
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
        child = _make_mock_agent(child_did)
        child._persisted_spawn_mandate = mandate
        manager._agents[child_name] = child
        manager._agent_names[child_did] = child_name
        manager._parent_children[parent_did] = [child_name]
        manager._child_mandates[child_name] = mandate
        manager._created_configs[child_name] = config

        assert manager.begin_terminal_spawn_retirement(
            child_name,
            expected_child_did=child_did,
        ) is True
        witness = registry.get(child_did)
        assert witness is not None and witness.state == "retiring"
        assert (
            manager._reconcile_spawn_authority_restart_roster(
                MultiAgentConfig(agents={})
            ).agents
            == {}
        )
        with pytest.raises(RuntimeError, match="denied restart"):
            manager._verify_agent_authority(child_name, child)

        assert manager.cancel_terminal_spawn_retirement(
            child_name,
            expected_child_did=child_did,
        ) is True
        witness = registry.get(child_did)
        assert witness is not None and witness.active
        assert child_name in manager._reconcile_spawn_authority_restart_roster(
            MultiAgentConfig(agents={})
        ).agents

    @pytest.mark.parametrize("terminal_state", ["retiring", "retired"])
    def test_terminal_host_witness_removes_exact_stale_startup_row(
        self,
        tmp_path,
        terminal_state,
    ):
        child_name = "TerminalRosterChild"
        child_did = "did:test:terminal-roster-child"
        config_path = tmp_path / "multi_agent.toml"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        MultiAgentConfig(agents={child_name: local}).save(config_path)
        private_key, _ = generate_secp256k1_keypair()
        mandate = sign_mandate(
            SpawnMandate(
                parent_did="did:test:terminal-roster-parent",
                child_did=child_did,
            ),
            private_key,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=local,
        )
        registry.begin_retirement(child_name=child_name, child_did=child_did)
        if terminal_state == "retired":
            registry.retire(child_name=child_name, child_did=child_did)
        manager = AgentManager(
            base_data_dir=tmp_path,
            startup_config_path=config_path,
        )

        with patch(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did_sync",
            return_value=child_did,
        ) as read_anchor:
            reconciled = manager._reconcile_spawn_authority_restart_roster(
                MultiAgentConfig.from_file(config_path)
            )

        assert reconciled.agents == {}
        assert MultiAgentConfig.from_file(config_path).agents == {}
        read_anchor.assert_called_once_with(
            str(local.resolve_data_dir(tmp_path)),
            mode=AgentDIDLookupMode.INSPECTION,
        )

    def test_terminal_host_witness_preserves_replacement_identity_row(self, tmp_path):
        child_name = "ReplacedTerminalChild"
        retired_did = "did:test:retired-roster-child"
        replacement_did = "did:test:replacement-roster-child"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        private_key, _ = generate_secp256k1_keypair()
        mandate = sign_mandate(
            SpawnMandate(
                parent_did="did:test:retired-roster-parent",
                child_did=retired_did,
            ),
            private_key,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=child_name,
            child_did=retired_did,
            mandate=mandate,
            config=local,
        )
        registry.retire(child_name=child_name, child_did=retired_did)
        manager = AgentManager(base_data_dir=tmp_path)
        roster = MultiAgentConfig(agents={child_name: local})

        with patch(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did_sync",
            return_value=replacement_did,
        ):
            reconciled = manager._reconcile_spawn_authority_restart_roster(roster)

        assert reconciled.agents == roster.agents

    @pytest.mark.asyncio
    async def test_terminal_host_witness_inspection_failure_isolated_to_agent_load(
        self,
        tmp_path,
    ):
        """An unavailable terminal child slot cannot withhold healthy tenants."""

        child_name = "UnavailableTerminalChild"
        child_did = "did:test:unavailable-terminal-child"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
        )
        healthy = LocalAgentConfig(
            data_dir=Path("agent_data") / "HealthyPeer",
            port=8803,
        )
        private_key, _ = generate_secp256k1_keypair()
        mandate = sign_mandate(
            SpawnMandate(
                parent_did="did:test:terminal-roster-parent",
                child_did=child_did,
            ),
            private_key,
        )
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=local,
        )
        registry.retire(child_name=child_name, child_did=child_did)
        manager = AgentManager(base_data_dir=tmp_path)
        roster = MultiAgentConfig(
            agents={child_name: local, "HealthyPeer": healthy}
        )
        healthy_agent = _make_mock_agent("did:test:healthy-terminal-peer")

        async def initialize(name, _config):
            if name == child_name:
                raise OSError("terminal child database is unavailable")
            return healthy_agent

        with patch(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did_sync",
            side_effect=OSError("terminal child database is unavailable"),
        ), patch.object(manager, "_initialize_agent", side_effect=initialize):
            loaded = await manager.load_from_config(roster)

        assert loaded == 1
        assert manager.get_agent("HealthyPeer") is healthy_agent
        assert len(manager.init_failures) == 1
        failed_name, failure = manager.init_failures[0]
        assert failed_name == child_name
        assert isinstance(failure, OSError)

    @pytest.mark.asyncio
    async def test_expired_host_witness_inspection_failure_does_not_abort_fleet(
        self,
        tmp_path,
    ):
        """An unreadable expired child stays denied while healthy peers boot."""

        child_name = "UnreadableExpiredChild"
        child_did = "did:test:unreadable-expired-child"
        local = LocalAgentConfig(
            data_dir=Path("agent_data") / child_name,
            port=8802,
            autostart=False,
        )
        healthy = LocalAgentConfig(
            data_dir=Path("agent_data") / "HealthyExpiredPeer",
            port=8803,
        )
        data_dir = local.resolve_data_dir(tmp_path)
        data_dir.mkdir(parents=True)
        (data_dir / "kestrel_prime.db").touch()
        registry = SpawnAuthorityRegistry(tmp_path)
        registry.record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=SpawnMandate(
                parent_did="did:test:unreadable-expired-parent",
                child_did=child_did,
                ttl_seconds=1,
                created_at=(
                    datetime.now(timezone.utc) - timedelta(seconds=10)
                ).isoformat(),
                parent_signature="signed-unreadable-expired-child",
            ),
            config=local,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        roster = MultiAgentConfig(
            agents={child_name: local, "HealthyExpiredPeer": healthy}
        )
        healthy_agent = _make_mock_agent("did:test:healthy-expired-peer")

        async def initialize(name, _config):
            if name == child_name:
                raise OSError("expired child database is unavailable")
            return healthy_agent

        with patch(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did_sync",
            side_effect=OSError("expired child database is unavailable"),
        ), patch.object(manager, "_initialize_agent", side_effect=initialize):
            loaded = await manager.load_from_config(roster)

        assert loaded == 1
        assert manager.get_agent("HealthyExpiredPeer") is healthy_agent
        witness = registry.get(child_did)
        assert witness is not None and witness.state == "retiring"
        assert manager.init_failures == []

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.inception_service.create_kestrel_identity_async", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_spawn_passes_parent_did_to_create(
        self, mock_llm_cls, mock_agent_cls, mock_get_did, mock_inception, tmp_path
    ):
        """spawn_agent should pass parent's DID to create_agent for delegation."""
        mock_get_did.return_value = "did:spawned-child"
        mock_child = _make_mock_agent("did:spawned-child")
        mock_child._raw_storage = SimpleNamespace(
            graph=SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
        )
        mock_agent_cls.return_value = mock_child

        manager = AgentManager(base_data_dir=tmp_path)

        parent = _make_mock_agent("did:parent-xyz")
        parent._private_key, _ = generate_secp256k1_keypair()
        parent.identity = None

        mandate = SpawnMandate(
            parent_did="did:parent-xyz",
            purpose="test spawn",
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager.spawn_agent("SpawnedBot", parent, mandate)

        # Verify inception received parent_did
        call_kwargs = mock_inception.call_args[1]
        assert call_kwargs["parent_did"] == "did:parent-xyz"

        # Verify parent-child tracking
        assert "SpawnedBot" in manager.get_children("did:parent-xyz")
        persisted = manager.get_mandate("SpawnedBot")
        assert persisted is not mandate
        assert persisted.child_did == "did:spawned-child"
        assert mandate.child_did is None

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        new_callable=AsyncMock,
    )
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_spawn_persists_signature_bound_to_final_child_did(
        self, mock_llm_cls, mock_agent_cls, mock_get_did, mock_inception, tmp_path
    ):
        from kestrel_sovereign.spawn.mandate import verify_mandate

        private_key, public_key = generate_secp256k1_keypair()
        parent = _make_mock_agent("did:parent-signed")
        parent._private_key = private_key
        parent.identity = None
        parent.features = {}
        graph = SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
        child = _make_mock_agent("did:child-final")
        runtime_projection = SpawnMandate(
            parent_did=parent.agent_id,
            child_did=child.agent_id,
            features_allowed=[],
        )
        child.spawn_mandate = runtime_projection
        child._raw_storage = SimpleNamespace(graph=graph)
        mock_get_did.return_value = child.agent_id
        mock_agent_cls.return_value = child
        manager = AgentManager(base_data_dir=tmp_path)
        mandate = SpawnMandate(parent_did=parent.agent_id, purpose="signed")

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager.spawn_agent("SignedChild", parent, mandate)

        persisted = child._persisted_spawn_mandate
        assert mandate.child_did is None
        assert persisted is not mandate
        assert persisted.child_did == child.agent_id
        assert verify_mandate(persisted, public_key)
        graph.add_trusted_cross_agent_edge.assert_awaited_once_with(
            child.agent_id,
            parent.agent_id,
            "spawned_by",
            properties=persisted.to_edge_properties(),
        )
        assert child._persisted_spawn_mandate is persisted
        assert child.spawn_mandate is runtime_projection
        assert manager._spawn_authority_registry.pending() == ()
        witness = manager._spawn_authority_registry.get(child.agent_id)
        assert witness is not None and witness.active

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.inception_service.create_kestrel_identity_async", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_spawn_wires_mandate_features_into_child(
        self, mock_llm_cls, mock_agent_cls, mock_get_did, mock_inception, tmp_path
    ):
        """The mandate's feature allowlist must reach the child KestrelAgent.

        Regression for #1946: spawn validated ``features_allowed`` but never
        threaded it into the child's config, so ``load_agent`` built the child
        with ``allowed_features=None`` and it loaded ALL features regardless of
        what the mandate permitted. This drives the real
        spawn_agent -> _do_spawn -> create_agent -> load_agent chain (only
        inception/DID/KestrelAgent/LLMService are mocked) and asserts the child
        is constructed with the allowlist as ``allowed_features``.
        """
        mock_get_did.return_value = "did:featured-child"
        mock_child = _make_mock_agent("did:featured-child")
        mock_child._raw_storage = SimpleNamespace(
            graph=SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
        )
        mock_agent_cls.return_value = mock_child

        manager = AgentManager(base_data_dir=tmp_path)

        parent = _make_mock_agent("did:parent-feat")
        parent._private_key, _ = generate_secp256k1_keypair()
        parent.identity = None

        mandate = SpawnMandate(
            parent_did="did:parent-feat",
            purpose="restricted child",
            features_allowed=["MemoryFeature", "WebSearchFeature"],
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager.spawn_agent("FeaturedBot", parent, mandate)

        # The child KestrelAgent must be built WITH the allowlist, not None.
        assert mock_agent_cls.call_count == 1
        child_kwargs = mock_agent_cls.call_args.kwargs
        assert child_kwargs["allowed_features"] == {"MemoryFeature", "WebSearchFeature"}

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.inception_service.create_kestrel_identity_async", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_spawn_without_features_loads_all(
        self, mock_llm_cls, mock_agent_cls, mock_get_did, mock_inception, tmp_path
    ):
        """An empty (default) mandate allowlist means "load all" (allowed_features=None)."""
        mock_get_did.return_value = "did:open-child"
        mock_child = _make_mock_agent("did:open-child")
        mock_child._raw_storage = SimpleNamespace(
            graph=SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
        )
        mock_agent_cls.return_value = mock_child

        manager = AgentManager(base_data_dir=tmp_path)

        parent = _make_mock_agent("did:parent-open")
        parent._private_key, _ = generate_secp256k1_keypair()
        parent.identity = None

        # No features_allowed → default empty list → load all features.
        mandate = SpawnMandate(parent_did="did:parent-open", purpose="open child")

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager.spawn_agent("OpenBot", parent, mandate)

        assert mock_agent_cls.call_count == 1
        assert mock_agent_cls.call_args.kwargs["allowed_features"] is None

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.inception_service.create_kestrel_identity_async", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_spawn_duplicate_name_raises(
        self, mock_llm_cls, mock_agent_cls, mock_get_did, mock_inception, tmp_path
    ):
        """spawn_agent should fail if child name already exists."""
        manager = AgentManager(base_data_dir=tmp_path)
        manager._agents["Existing"] = _make_mock_agent("did:existing")

        parent = _make_mock_agent("did:parent")
        parent._private_key = None
        parent.identity = None

        mandate = SpawnMandate(parent_did="did:parent", purpose="dupe test")

        with pytest.raises(ValueError, match="already exists"):
            await manager.spawn_agent("Existing", parent, mandate)
