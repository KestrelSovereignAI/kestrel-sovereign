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
    FeatureContributionRuntimeError,
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
    prepared = runtime.prepare_transition((feature,)).only()

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
    prepared = runtime.prepare_transition((feature,)).only()
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


# ---------------------------------------------------------------------------
# Blast radius of a contribution collision (issue #2951)
# ---------------------------------------------------------------------------


class _PeerFixtureFeature(SDKFixtureFeature):
    """A second feature with its own owner and its own contribution keys."""

    contribution_prefix = "peer-fixture"


def _rival_source(source):
    """Same NAME, genuinely different contract.

    `trust` is part of the contract signature, so this is a mismatch by the
    registry's own definition — no reliance on closure-identity subtleties.
    """
    import dataclasses

    from kestrel_sdk.signals import Trust

    other = Trust.UNTRUSTED if source.trust is Trust.TRUSTED else Trust.TRUSTED
    return dataclasses.replace(source, trust=other)


def test_a_clash_with_an_already_registered_source_rejects_only_that_feature(tmp_path):
    """One stale feature must not abort boot for the whole host.

    The observed outage: `kestrel-feature-talon` 0.2.0 still contributed a
    source core had just reclaimed, and EVERY agent on the host failed to load.
    The collision is a capability gap for that feature, not an identity gap for
    the agent.
    """
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    peer = _PeerFixtureFeature(agent)
    runtime = agent._ensure_feature_contribution_runtime()

    incumbent = _rival_source(feature.source)
    # The premise, asserted rather than assumed: same name, different contract.
    assert incumbent.name == feature.source.name
    assert not SourceRegistry.contract_equivalent(incumbent, feature.source)
    runtime.source_registry.register(incumbent)

    transition = runtime.prepare_transition((feature, peer))

    # The clashing feature is refused, and the reason names the source...
    assert [r.feature for r in transition.rejected] == [feature]
    assert feature.source.name in transition.rejected[0].reason
    # ...its peer is unaffected, and the transition itself did not fail.
    assert [item.feature for item in transition.accepted] == [peer]
    assert [f for f, _ in transition.activatable((feature, peer))] == [peer]
    # The incumbent registration is left exactly as it was.
    assert runtime.source_registry.get(feature.source.name) is incumbent


def test_an_equivalent_duplicate_contribution_is_a_no_op_success(tmp_path):
    """Byte-identical re-registration is what every declared policy calls fine.

    The old preflight compared NAMES, so the common case during an extraction —
    core and a feature shipping the same source — was fatal. The registry
    already knows what "the same source" means, and it is not the name.
    """
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    runtime = agent._ensure_feature_contribution_runtime()

    runtime.source_registry.register(feature.source)
    assert SourceRegistry.contract_equivalent(
        runtime.source_registry.get(feature.source.name), feature.source
    )

    transition = runtime.prepare_transition((feature,))

    assert transition.rejected == ()
    assert [item.feature for item in transition.accepted] == [feature]


def test_two_features_claiming_one_name_in_the_same_batch_still_raises(tmp_path):
    """Transition-invalid is a different class from feature-rejected.

    Nothing is already registered here — the PROPOSED set is incoherent, and
    there is no principled subset to prefer, so this must keep raising rather
    than silently dropping whichever feature happens to be second.
    """
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    twin = _PeerFixtureFeature(agent)
    twin.workflow_registration = type(twin.workflow_registration)(
        owner=twin.contribution_owner,
        name=twin.workflow_registration.name,
        actor=twin.actor,
        sources=(feature.source,),   # the SAME source name, from both features
    )

    runtime = agent._ensure_feature_contribution_runtime()
    assert runtime.source_registry.get(feature.source.name) is None  # premise

    with pytest.raises(FeatureContributionRuntimeError, match="workflow source name"):
        runtime.prepare_transition((feature, twin))


def test_activating_a_single_rejected_feature_still_raises(tmp_path):
    """`only()` is the one-feature path: enabling a named feature must fail.

    There is no fleet to keep up by continuing — the caller asked for exactly
    this feature — so a rejection surfaces as that operation's failure rather
    than as an empty accepted set the caller would have to notice.
    """
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    runtime = agent._ensure_feature_contribution_runtime()
    runtime.source_registry.register(_rival_source(feature.source))

    with pytest.raises(FeatureContributionRuntimeError, match="different contract"):
        runtime.prepare_transition((feature,)).only()


def test_health_reports_a_refused_feature_and_refuses_to_call_it_healthy():
    """A missing capability the operator thinks they have is not `healthy`.

    Logging the rejection at boot files it where nobody is looking by the time
    it matters. `/health/detailed` is the surface that has to carry it.
    """
    import types

    from kestrel_sovereign.features.contribution_runtime import ContributionRejection
    from kestrel_sovereign.server import _with_contribution_rejections

    agent = types.SimpleNamespace(
        rejected_feature_contributions=(
            ContributionRejection(object(), "TalonFeature", "signal source ..."),
        )
    )
    original = {"status": "healthy", "checks": []}

    merged = _with_contribution_rejections(agent, original)

    assert merged["status"] == "degraded"
    assert merged["features_not_loaded"] == [
        {"feature": "TalonFeature", "reason": "signal source ..."}
    ]
    # The health feature's cached dict is shared; it must not be mutated.
    assert original == {"status": "healthy", "checks": []}


def test_health_is_unchanged_when_nothing_was_refused():
    """No rejections must not manufacture a key or downgrade a healthy host."""
    import types

    from kestrel_sovereign.server import _with_contribution_rejections

    agent = types.SimpleNamespace(rejected_feature_contributions=())
    original = {"status": "healthy", "checks": []}

    assert _with_contribution_rejections(agent, original) is original


def test_deactivating_a_feature_does_not_remove_an_equivalent_incumbent(tmp_path):
    """An equivalent contribution kept the INCUMBENT — it is not ours to remove.

    Accepting equivalent duplicates created this: `activate` recorded every
    DECLARED source as registered, so teardown either unregistered core's own
    source or failed identity validation when the objects merely matched by
    contract. Only what a lifecycle newly added is its to tear down.
    """
    from tests.fixtures.sdk_contribution_fixture import _source

    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    runtime = agent._ensure_feature_contribution_runtime()

    # A DISTINCT but contract-equivalent incumbent — core registered its own
    # object, which is the shape that made teardown raise identity mismatch.
    incumbent = _source(feature.source.name)
    assert incumbent is not feature.source
    assert SourceRegistry.contract_equivalent(incumbent, feature.source)
    runtime.source_registry.register(incumbent)

    prepared = runtime.prepare_transition((feature,)).only()
    runtime.activate(prepared)

    assert runtime.deactivate(feature) is True
    # The incumbent survives the feature's teardown, unchanged.
    assert runtime.source_registry.get(feature.source.name) is incumbent


def test_a_setup_step_name_clash_rejects_the_feature_not_the_boot(tmp_path):
    """`preflight` raises on an already-registered NAME too, not just on order.

    Gating the per-feature setup-step check on the non-order path left exactly
    the blast radius this split removes — intact for one key type: one feature's
    setup-step collision still aborted every agent's boot.
    """
    agent = _agent(tmp_path)
    feature = SDKFixtureFeature(agent)
    peer = _PeerFixtureFeature(agent)
    runtime = agent._ensure_feature_contribution_runtime()

    # Something already owns that setup-step name.
    runtime.setup_step_registry.register_batch((feature.setup_registration,))

    transition = runtime.prepare_transition((feature, peer))

    assert [r.feature for r in transition.rejected] == [feature]
    assert "setup step already registered" in transition.rejected[0].reason
    assert [item.feature for item in transition.accepted] == [peer]


def test_host_feature_rejections_reach_detailed_health():
    """A refused HOST feature is agent-scoped nowhere, so nothing would show it.

    It was recorded on the host context and never read: the reason lived only in
    boot logs while the endpoint reported healthy over skipped host routes.
    """
    import types

    from kestrel_sovereign.features.contribution_runtime import ContributionRejection
    from kestrel_sovereign.server import _with_host_feature_rejections

    state = types.SimpleNamespace(
        host_context=types.SimpleNamespace(
            rejected_host_feature_contributions=(
                ContributionRejection(object(), "EyeHostFeature", "wait provider ..."),
            )
        )
    )
    merged = _with_host_feature_rejections(state, {"status": "healthy", "checks": []})

    assert merged["status"] == "degraded"
    assert merged["host_features_not_loaded"] == [
        {"feature": "EyeHostFeature", "reason": "wait provider ..."}
    ]


def test_host_feature_health_is_unchanged_with_no_host_context():
    """No host context must not manufacture a key or downgrade a healthy host."""
    import types

    from kestrel_sovereign.server import _with_host_feature_rejections

    payload = {"status": "healthy", "checks": []}
    assert _with_host_feature_rejections(types.SimpleNamespace(), payload) is payload


def test_tearing_down_one_feature_keeps_a_source_another_still_declares(tmp_path):
    """A shared source must outlive the first holder's teardown.

    Feature B activating against an equivalent incumbent records no ownership,
    so once A stopped owning everything it declared, tearing A down removed the
    only registration while B's workflow was still dispatching against it. The
    lease is derived from what is still active, not counted in the registry.
    """
    agent = _agent(tmp_path)
    first = SDKFixtureFeature(agent)
    second = _PeerFixtureFeature(agent)
    # `second` declares a source EQUIVALENT to `first`'s, under the same name.
    second.workflow_registration = type(second.workflow_registration)(
        owner=second.contribution_owner,
        name=second.workflow_registration.name,
        actor=second.actor,
        sources=(first.source,),
    )
    runtime = agent._ensure_feature_contribution_runtime()

    # Activated separately — together they would be a within-batch duplicate.
    runtime.activate(runtime.prepare_transition((first,)).only())
    runtime.activate(runtime.prepare_transition((second,)).only())
    assert runtime.source_registry.get(first.source.name) is first.source

    runtime.deactivate(first)

    # `second` is still active and still dispatches against that source.
    assert runtime.is_active(second)
    assert runtime.source_registry.get(first.source.name) is first.source

    # Once the last holder goes, so does the registration.
    runtime.deactivate(second)
    assert runtime.source_registry.get(first.source.name) is None


def test_a_failed_teardown_leaves_the_holders_unchanged(tmp_path):
    """A raise mid-teardown must not change who holds the source.

    There is no ownership transfer to get wrong any more — the registry holds
    the claims — but a teardown that raises must still leave the claim list
    exactly as it was, and a retry must not accumulate anything.
    """
    agent = _agent(tmp_path)
    first = SDKFixtureFeature(agent)
    second = _PeerFixtureFeature(agent)
    second.workflow_registration = type(second.workflow_registration)(
        owner=second.contribution_owner,
        name=second.workflow_registration.name,
        actor=second.actor,
        sources=(first.source,),
    )
    runtime = agent._ensure_feature_contribution_runtime()
    runtime.activate(runtime.prepare_transition((first,)).only())
    runtime.activate(runtime.prepare_transition((second,)).only())

    name = first.source.name
    before = runtime.source_registry.owners_of(name)
    assert set(before) == {first, second}   # premise: both hold it

    # Break an UNRELATED inverse so teardown validation raises.
    runtime.wait_registry.unregister(first.wait_provider.kind)

    for _ in range(2):
        with pytest.raises(FeatureContributionRuntimeError, match="wait-provider"):
            runtime.deactivate(first)
        assert runtime.source_registry.owners_of(name) == before
        assert runtime.source_registry.get(name) is first.source


def test_the_registry_is_the_only_ownership_ledger(tmp_path):
    """A feature that BOTH self-registers and declares a source, torn down
    while another feature still holds it.

    This is the shape neither ledger could see: `Feature._own_signal_sources`
    recorded the imperative claim, the contribution runtime recorded the
    declarative one, and each tore down against its own list. The first
    feature's shutdown removed a source the second was still dispatching
    against. One ledger in the registry makes it unrepresentable.
    """
    agent = _agent(tmp_path)
    first = SDKFixtureFeature(agent)
    second = _PeerFixtureFeature(agent)
    second.workflow_registration = type(second.workflow_registration)(
        owner=second.contribution_owner,
        name=second.workflow_registration.name,
        actor=second.actor,
        sources=(first.source,),
    )
    runtime = agent._ensure_feature_contribution_runtime()
    agent.signal_registry = runtime.source_registry

    # `first` registers the source ITSELF, then records the claim — the two
    # calls `Feature._own_signal_sources` makes (the fixture is not a Feature
    # subclass, so the registry API it drives is exercised directly here; the
    # Feature wiring itself is covered in test_feature_runtime_lifecycle.py).
    runtime.source_registry.register_with_policy(first.source)
    runtime.source_registry.adopt(first.source.name, first, created=True)
    # ...and also declares it (the declarative path).
    runtime.activate(runtime.prepare_transition((first,)).only())
    runtime.activate(runtime.prepare_transition((second,)).only())

    name = first.source.name
    assert set(runtime.source_registry.owners_of(name)) == {first, second}

    # Tear `first` down through BOTH paths, in the order a real disable uses.
    runtime.deactivate(first)
    runtime.source_registry.release_all(first)   # what shutdown() does

    # `second` is still active, so the source is still there for it.
    assert runtime.is_active(second)
    assert runtime.source_registry.get(name) is first.source
    assert runtime.source_registry.owners_of(name) == (second,)

    # And when the last holder goes, so does the source.
    runtime.deactivate(second)
    assert runtime.source_registry.get(name) is None
