from __future__ import annotations

import traceback
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kestrel_sdk.features import ContributionContractError
from kestrel_sdk.operator import ExecutionTargetDescriptor

from kestrel_sovereign import server
from kestrel_sovereign.features import MandatoryFeatureReadinessError
from kestrel_sovereign.features.contribution_runtime import (
    FeatureContributionCollectionError,
)
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


def test_owner_validation_failure_is_typed_and_identifies_incoming_owner(tmp_path):
    agent = _agent(tmp_path)

    class SameOwnerFixture(SDKFixtureFeature):
        @property
        def contribution_owner(self):
            return "tests:shared-owner"

    first = SameOwnerFixture(agent)
    second = SameOwnerFixture(agent)
    runtime = agent._ensure_feature_contribution_runtime()

    with pytest.raises(FeatureContributionCollectionError) as exc_info:
        runtime.prepare_transition((first, second))

    error = exc_info.value
    assert error.feature is second
    assert error.stage == "contribution validation"
    assert error.getter == "validate_contribution_owner_uniqueness"
    assert isinstance(error.__cause__, ContributionContractError)
    assert runtime.active_owners() == ()


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "getter",
    ["get_tools", "get_service_registrations"],
)
async def test_non_mandatory_collection_failure_remains_actionable(
    tmp_path, monkeypatch, getter
):
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    original = RuntimeError(f"actionable {getter} diagnostic")

    def fail_collection():
        raise original

    monkeypatch.setattr(feature, getter, fail_collection)

    with pytest.raises(RuntimeError, match=f"actionable {getter}") as exc_info:
        await agent._register_feature(feature)

    assert exc_info.value is original
    assert feature.name not in agent.features
    assert agent._ensure_feature_contribution_runtime().active_owners() == ()


@pytest.mark.asyncio
async def test_non_mandatory_collection_failure_preserves_original_nested_cause(
    tmp_path, monkeypatch
):
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    root = ValueError("root")
    original = RuntimeError("outer")

    def fail_collection():
        raise original from root

    monkeypatch.setattr(feature, "get_tools", fail_collection)

    with pytest.raises(RuntimeError, match="outer") as exc_info:
        await agent._register_feature(feature)

    assert exc_info.value is original
    assert exc_info.value.__cause__ is root
    visible_chain = "".join(traceback.format_exception(exc_info.value))
    assert "ValueError: root" in visible_chain
    assert "FeatureContributionCollectionError" not in visible_chain
    assert feature.name not in agent.features
    assert agent._ensure_feature_contribution_runtime().active_owners() == ()


@pytest.mark.asyncio
async def test_mandatory_non_tool_collection_failure_uses_contribution_diagnostic(
    tmp_path,
):
    agent = _agent(tmp_path)
    original = ContributionContractError("secret service diagnostic")

    class SecurityFeature(SDKFixtureFeature):
        def get_service_registrations(self):
            raise original

    feature = SecurityFeature(agent)

    with pytest.raises(MandatoryFeatureReadinessError) as exc_info:
        await agent._register_feature(feature)

    error = exc_info.value
    assert error.stage == "contribution registration"
    assert error.problem == "could not register its SDK contributions"
    assert error.stage != "registration"
    assert "tools" not in str(error)
    assert "secret service diagnostic" not in str(error)
    assert error.__cause__ is original
    assert feature.name not in agent.features
    assert agent._ensure_feature_contribution_runtime().active_owners() == ()


def test_tool_name_projection_uses_tool_collection_boundary(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    monkeypatch.setattr(feature, "get_tools", lambda: (object(),))
    runtime = agent._ensure_feature_contribution_runtime()

    with pytest.raises(FeatureContributionCollectionError) as exc_info:
        runtime.prepare_transition((feature,))

    error = exc_info.value
    assert error.feature is feature
    assert error.stage == "tool collection"
    assert error.getter == "get_tools"
    assert "has no attribute" not in str(error)
    assert isinstance(error.__cause__, AttributeError)
    assert runtime.active_owners() == ()


@pytest.mark.asyncio
async def test_mandatory_tool_name_projection_failure_uses_tool_diagnostic(tmp_path):
    agent = _agent(tmp_path)

    class SecurityFeature(SDKFixtureFeature):
        def get_tools(self):
            return (object(),)

    feature = SecurityFeature(agent)

    with pytest.raises(MandatoryFeatureReadinessError) as exc_info:
        await agent._register_feature(feature)

    error = exc_info.value
    assert error.stage == "registration"
    assert error.problem == "could not register its tools"
    assert "has no attribute" not in str(error)
    assert isinstance(error.__cause__, AttributeError)
    assert feature.name not in agent.features
    assert agent._ensure_feature_contribution_runtime().active_owners() == ()


@pytest.mark.parametrize(
    ("getter", "expected_stage"),
    [
        ("get_tools", "tool collection"),
        ("get_service_registrations", "service collection"),
        ("get_wait_provider_registrations", "wait-provider collection"),
        ("get_workflow_registrations", "workflow collection"),
        ("get_feature_permission_defaults", "permission-default collection"),
        ("get_setup_step_registrations", "setup-step collection"),
    ],
)
def test_every_contribution_getter_uses_one_sanitized_typed_boundary(
    tmp_path, monkeypatch, getter, expected_stage
):
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    runtime = agent._ensure_feature_contribution_runtime()
    original = RuntimeError(f"credential-from-{getter}")

    def fail_collection():
        raise original

    monkeypatch.setattr(feature, getter, fail_collection)

    with pytest.raises(FeatureContributionCollectionError) as exc_info:
        runtime.prepare_transition((feature,))

    error = exc_info.value
    assert error.feature is feature
    assert error.getter == getter
    assert error.stage == expected_stage
    assert f"credential-from-{getter}" not in str(error)
    assert error.__cause__ is original
    assert runtime.active_owners() == ()


def test_contribution_validation_uses_same_sanitized_typed_boundary(
    tmp_path, monkeypatch
):
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    runtime = agent._ensure_feature_contribution_runtime()
    original = ContributionContractError("token=validation-secret")

    def fail_validation(*args, **kwargs):
        raise original

    monkeypatch.setattr(
        "kestrel_sovereign.features.contribution_runtime."
        "validate_feature_contributions",
        fail_validation,
    )

    with pytest.raises(FeatureContributionCollectionError) as exc_info:
        runtime.prepare_transition((feature,))

    error = exc_info.value
    assert error.feature is feature
    assert error.getter == "validate_feature_contributions"
    assert error.stage == "contribution validation"
    assert "validation-secret" not in str(error)
    assert error.__cause__ is original
    assert runtime.active_owners() == ()


def test_collection_error_rejects_arbitrary_public_boundary_text():
    feature = object()
    error = FeatureContributionCollectionError(
        feature,
        "secret-getter=/private/credentials",
    )

    assert error.feature is feature
    assert error.getter == "unknown contribution boundary"
    assert error.stage == "contribution collection"
    assert "secret-getter" not in str(error)
    assert "/private/credentials" not in str(error)


def test_batch_getter_failure_leaves_every_collected_owner_uncommitted(
    tmp_path, monkeypatch
):
    agent = _agent(tmp_path)
    first = SDKFixtureFeature(agent)

    class SecondFixture(SDKFixtureFeature):
        name = "second-fixture"
        contribution_prefix = "second-fixture"

    second = SecondFixture(agent)
    original = RuntimeError("secret second workflow failure")

    def fail_collection():
        raise original

    monkeypatch.setattr(second, "get_workflow_registrations", fail_collection)
    runtime = agent._ensure_feature_contribution_runtime()

    with pytest.raises(FeatureContributionCollectionError) as exc_info:
        runtime.prepare_transition((first, second))

    assert exc_info.value.feature is second
    assert exc_info.value.__cause__ is original
    assert runtime.active_owners() == ()
    assert len(agent.wait_registry.kinds()) == 0
    assert len(agent.signal_registry) == 0


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
