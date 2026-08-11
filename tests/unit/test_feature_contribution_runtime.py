from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kestrel_sdk.features import ContributionContractError
from kestrel_sdk.operator import ExecutionTargetDescriptor

from kestrel_sovereign import server
from kestrel_sovereign.features import MandatoryFeatureReadinessError
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.operator import (
    ExecutionTargetRegistration,
    OperatorRegistrationIdentityError,
    OperatorRuntimeRegistry,
)
from kestrel_sovereign.signals import SourceRegistry
from kestrel_sovereign.waits import WaitRegistry
from kestrel_sovereign.ui_contributions import compute_ui_manifest
from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature


def _agent(tmp_path: Path) -> KestrelAgent:
    agent = KestrelAgent(
        did="did:test:sdk-contributions",
        storage_path=str(tmp_path / "agent.db"),
    )
    agent.task_manager = None
    agent.signal_registry = SourceRegistry()
    agent.wait_registry = WaitRegistry()
    agent.operator_registry = OperatorRuntimeRegistry()
    agent.feature_contribution_runtime = None
    agent.features = {}
    return agent


def _assert_live(agent, feature, expected: bool) -> None:
    assert (
        agent.wait_registry.get(feature.wait_provider.kind) is feature.wait_provider
    ) is expected
    assert (agent.signal_registry.get(feature.source.name) is feature.source) is expected
    assert (
        agent.operator_registry.resolve_service(
            feature.service_registration.reference
        )
        is feature.service
    ) is expected
    assert (
        agent.operator_registry.resolve_workflow_actor(
            feature.workflow_registration.name
        )
        is feature.actor
    ) is expected
    assert (
        agent.permission_defaults_registry.get(feature.name)
        is feature.permission_defaults
    ) is expected
    assert (
        agent.setup_step_registry.get(feature.setup_registration.name)
        is feature.setup_registration
    ) is expected


@pytest.mark.asyncio
async def test_agent_lifecycle_registers_exact_sdk_contributions_once(tmp_path):
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)

    await agent._register_feature(feature)

    assert feature.contribution_calls == {
        "services": 1,
        "waits": 1,
        "workflows": 1,
        "permissions": 1,
        "setup": 1,
    }
    _assert_live(agent, feature, True)

    await agent._unregister_feature_runtime(feature, unload=False)
    _assert_live(agent, feature, False)

    await agent._activate_feature_runtime(feature)
    assert set(feature.contribution_calls.values()) == {2}
    _assert_live(agent, feature, True)

    await agent._unregister_feature_runtime(feature, unload=True)
    _assert_live(agent, feature, False)


@pytest.mark.asyncio
async def test_owner_conflict_rejects_complete_transition_before_mutation(tmp_path):
    agent = _agent(tmp_path)

    class SameOwnerFixture(SDKFixtureFeature):
        @property
        def contribution_owner(self):
            return "tests:shared-owner"

    first = SameOwnerFixture(agent)
    second = SameOwnerFixture(agent)

    with pytest.raises(
        ContributionContractError,
        match="duplicate active feature contribution_owner",
    ):
        agent._prepare_feature_contribution_transition((first, second))

    assert not first.initialized
    assert not second.initialized
    assert len(agent.wait_registry.kinds()) == 0
    assert len(agent.signal_registry) == 0


@pytest.mark.asyncio
async def test_activation_failure_reverses_declarative_contributions(tmp_path):
    agent = _agent(tmp_path)

    class FailingFixture(SDKFixtureFeature):
        async def on_enable(self):
            raise RuntimeError("enable failed")

    feature = FailingFixture(agent)
    with pytest.raises(RuntimeError, match="enable failed"):
        await agent._register_feature(feature)

    _assert_live(agent, feature, False)
    assert feature.name not in agent.features


@pytest.mark.asyncio
async def test_mandatory_contribution_failure_has_actionable_sanitized_diagnostic(
    tmp_path, monkeypatch
):
    agent = _agent(tmp_path)

    class SecurityFeature(SDKFixtureFeature):
        pass

    feature = SecurityFeature(agent)
    runtime = agent._ensure_feature_contribution_runtime()

    def fail_activation(_prepared):
        raise RuntimeError("api-key=must-not-reach-health")

    monkeypatch.setattr(runtime, "activate", fail_activation)

    with pytest.raises(MandatoryFeatureReadinessError) as exc_info:
        await agent._register_feature(feature)

    error = exc_info.value
    assert error.feature_name == "SecurityFeature"
    assert error.stage == "contribution registration"
    assert error.problem == "could not register its SDK contributions"
    assert "could not register its SDK contributions" in str(error)
    assert "during contribution registration" in str(error)
    assert "api-key" not in str(error)
    assert "api-key" in str(error.__cause__)

    arbitrary = MandatoryFeatureReadinessError(
        "SecurityFeature",
        "secret-stage=/private/runtime",
        "secret-problem=credential",
    )
    assert arbitrary.stage == "readiness"
    assert arbitrary.problem == "failed"
    assert "secret-stage" not in str(arbitrary)
    assert "secret-problem" not in str(arbitrary)


def test_permission_registration_failure_preserves_original_error_and_cleans_peers(
    tmp_path, monkeypatch
):
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    runtime = agent._ensure_feature_contribution_runtime()
    prepared = runtime.prepare_transition((feature,))[0]

    def fail_register(_registration):
        raise ValueError("permission commit failed")

    def unexpected_unregister(_registration):
        raise AssertionError("uncommitted permission registration was rolled back")

    monkeypatch.setattr(runtime.permission_defaults_registry, "register", fail_register)
    monkeypatch.setattr(
        runtime.permission_defaults_registry,
        "unregister",
        unexpected_unregister,
    )

    with pytest.raises(ValueError, match="permission commit failed"):
        runtime.activate(prepared)

    assert runtime.wait_registry.get(feature.wait_provider.kind) is None
    assert runtime.source_registry.get(feature.source.name) is None
    assert (
        runtime.operator_registry.resolve_service(
            feature.service_registration.reference
        )
        is None
    )


def test_scoped_execution_targets_follow_feature_lifecycle_without_touching_peers(
    tmp_path,
):
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    runtime = agent._ensure_feature_contribution_runtime()
    prepared = runtime.prepare_transition((feature,))[0]
    runtime.activate(prepared)

    owned = ExecutionTargetRegistration(
        owner=feature.contribution_owner,
        descriptor=ExecutionTargetDescriptor(
            target_id="owned-target",
            target_kind="container",
            display_name="Owned target",
            tenant_id="tenant",
            boundary_id="workspace",
            capabilities=frozenset({"shell.execute"}),
        ),
        handle=object(),
    )
    peer = ExecutionTargetRegistration(
        owner="tests:peer-target-owner",
        descriptor=ExecutionTargetDescriptor(
            target_id="peer-target",
            target_kind="container",
            display_name="Peer target",
            tenant_id="tenant",
            boundary_id="workspace",
            capabilities=frozenset({"shell.execute"}),
        ),
        handle=object(),
    )
    owned_set = runtime.register_execution_targets(feature, (owned,))
    peer_set = runtime.operator_registry.register(
        peer.owner, execution_targets=(peer,)
    )

    runtime.deactivate(feature)

    with pytest.raises(OperatorRegistrationIdentityError):
        runtime.operator_registry.validate_registration_set(owned_set)
    runtime.operator_registry.validate_registration_set(peer_set)


@pytest.mark.asyncio
async def test_scoped_execution_target_rolls_back_when_feature_enable_fails(
    tmp_path,
):
    class TargetThenFailFeature(SDKFixtureFeature):
        async def on_enable(self):
            target = ExecutionTargetRegistration(
                owner=self.contribution_owner,
                descriptor=ExecutionTargetDescriptor(
                    target_id="rollback-target",
                    target_kind="container",
                    display_name="Rollback target",
                    tenant_id="tenant",
                    boundary_id="workspace",
                    capabilities=frozenset({"shell.execute"}),
                ),
                handle=object(),
            )
            self.target_set = (
                self.agent.feature_contribution_runtime.register_execution_targets(
                    self,
                    (target,),
                )
            )
            raise RuntimeError("enable failed after target registration")

    agent = _agent(tmp_path)
    feature = TargetThenFailFeature(agent)

    with pytest.raises(RuntimeError, match="enable failed after target registration"):
        await agent._register_feature(feature)

    with pytest.raises(OperatorRegistrationIdentityError):
        agent.operator_registry.validate_registration_set(feature.target_set)
    _assert_live(agent, feature, False)


@pytest.mark.asyncio
async def test_disable_and_remove_hide_fixture_routes_and_ui(tmp_path):
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    await agent._register_feature(feature)

    app = FastAPI()
    app.state.agent = agent
    server._mount_feature_ui_assets(app, agents=(agent,))
    server._mount_feature_routers(app, agents=(agent,))

    with TestClient(app) as client:
        assert client.get("/api/agent-fixture/fixture").status_code == 200
        assert compute_ui_manifest(agent)

        await agent._unregister_feature_runtime(feature, unload=False)
        assert client.get("/api/agent-fixture/fixture").status_code == 404
        assert compute_ui_manifest(agent) == []

        await agent._activate_feature_runtime(feature)
        assert client.get("/api/agent-fixture/fixture").status_code == 200
        assert compute_ui_manifest(agent)

        await agent._unregister_feature_runtime(feature, unload=True)
        assert client.get("/api/agent-fixture/fixture").status_code == 404
        assert compute_ui_manifest(agent) == []


def test_core_never_imports_sdk_fixture_implementation():
    root = Path(__file__).parents[2]
    fixture_module = "sdk_contribution_fixture"
    for path in (root / "kestrel_sovereign").rglob("*.py"):
        assert fixture_module not in path.read_text(encoding="utf-8")

    fixture_source = (
        root / "tests" / "fixtures" / "sdk_contribution_fixture.py"
    ).read_text(encoding="utf-8")
    assert "kestrel_sovereign" not in fixture_source
