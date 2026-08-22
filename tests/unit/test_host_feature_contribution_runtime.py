from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kestrel_sdk.features import ContributionContractError

from kestrel_sovereign.host_features import (
    SovereignHostContext,
    mount_host_feature_routers,
    mount_host_feature_ui,
    start_host_features,
    stop_host_features,
    unmount_host_features,
)
from kestrel_sovereign.features.contribution_runtime import (
    FeatureContributionCollectionError,
    FeatureContributionRuntimeError,
)
from tests.fixtures.sdk_contribution_fixture import SDKFixtureHostFeature


def _assert_live(ctx, feature, expected: bool) -> None:
    assert (ctx.wait_registry.get(feature.wait_provider.kind) is feature.wait_provider) is expected
    assert (ctx.signal_registry.get(feature.source.name) is feature.source) is expected
    assert (
        ctx.operator_registry.resolve_service(feature.service_registration.reference)
        is feature.service
    ) is expected
    assert (
        ctx.operator_registry.resolve_workflow_actor(
            feature.workflow_registration.name
        )
        is feature.actor
    ) is expected
    assert (
        ctx.permission_defaults_registry.get(feature.name)
        is feature.permission_defaults
    ) is expected
    assert (
        ctx.setup_step_registry.get(feature.setup_registration.name)
        is feature.setup_registration
    ) is expected


@pytest.mark.asyncio
async def test_host_start_stop_wires_all_exact_sdk_contributions_once():
    ctx = SovereignHostContext()
    feature = SDKFixtureHostFeature()

    started = await start_host_features([feature], ctx)

    assert started == [feature]
    assert feature.started
    assert set(feature.contribution_calls.values()) == {1}
    _assert_live(ctx, feature, True)

    await stop_host_features([feature], ctx)
    assert feature.stopped
    _assert_live(ctx, feature, False)


@pytest.mark.asyncio
async def test_host_stop_removes_only_requested_owner():
    class PeerHostFeature(SDKFixtureHostFeature):
        name = "peer-host-fixture"
        contribution_prefix = "peer-host-fixture"

    first = SDKFixtureHostFeature()
    peer = PeerHostFeature()
    ctx = SovereignHostContext()
    await start_host_features([first, peer], ctx)

    await stop_host_features([first], ctx)

    _assert_live(ctx, first, False)
    _assert_live(ctx, peer, True)
    assert ctx.started_host_features == (peer,)
    await stop_host_features([peer], ctx)


@pytest.mark.asyncio
async def test_host_start_failure_removes_only_failed_feature_contributions():
    class FailingHostFeature(SDKFixtureHostFeature):
        contribution_prefix = "failing-host-fixture"
        name = "failing-host-fixture"

        async def on_host_start(self, ctx):
            raise RuntimeError("host start failed")

    failing = FailingHostFeature()
    healthy = SDKFixtureHostFeature()
    ctx = SovereignHostContext()

    started = await start_host_features([failing, healthy], ctx)

    assert started == [healthy]
    _assert_live(ctx, failing, False)
    _assert_live(ctx, healthy, True)
    await stop_host_features(started, ctx)
    _assert_live(ctx, healthy, False)


@pytest.mark.asyncio
async def test_host_owner_conflict_fails_before_either_feature_starts():
    class SameOwnerHost(SDKFixtureHostFeature):
        @property
        def contribution_owner(self):
            return "tests:shared-host-owner"

    first = SameOwnerHost()
    first.contribution_prefix = "shared-host-one"
    second = SameOwnerHost()
    second.contribution_prefix = "shared-host-two"
    ctx = SovereignHostContext()

    with pytest.raises(FeatureContributionCollectionError) as exc_info:
        await start_host_features([first, second], ctx)

    assert exc_info.value.feature is second
    assert exc_info.value.getter == "validate_contribution_owner_uniqueness"
    assert isinstance(exc_info.value.__cause__, ContributionContractError)
    assert not first.started
    assert not second.started
    assert len(ctx.feature_contribution_runtime.wait_registry.kinds()) == 0
    assert len(ctx.feature_contribution_runtime.source_registry) == 0


@pytest.mark.asyncio
async def test_host_declarative_commit_failure_rejects_partial_startup(
    monkeypatch,
):
    class FirstHostFeature(SDKFixtureHostFeature):
        name = "first-host-fixture"
        contribution_prefix = "first-host-fixture"

    class SecondHostFeature(SDKFixtureHostFeature):
        name = "second-host-fixture"
        contribution_prefix = "second-host-fixture"

    first = FirstHostFeature()
    second = SecondHostFeature()
    resident = SDKFixtureHostFeature()
    ctx = SovereignHostContext()
    await start_host_features([resident], ctx)
    registry = ctx.feature_contribution_runtime.permission_defaults_registry
    original_register = registry.register

    def fail_second(registration):
        if registration.feature_name == second.name:
            raise ValueError("second permission commit failed")
        original_register(registration)

    monkeypatch.setattr(registry, "register", fail_second)

    with pytest.raises(
        FeatureContributionRuntimeError,
        match="contribution activation failed",
    ):
        await start_host_features([first, second], ctx)

    assert first.started and first.stopped
    assert not second.started
    _assert_live(ctx, first, False)
    _assert_live(ctx, second, False)
    _assert_live(ctx, resident, True)
    assert ctx.started_host_features == (resident,)
    await stop_host_features([resident], ctx)


@pytest.mark.asyncio
async def test_invalid_restart_candidate_preserves_already_valid_host_state():
    healthy = SDKFixtureHostFeature()
    ctx = SovereignHostContext()
    await start_host_features([healthy], ctx)

    class ConflictingRestartFeature(SDKFixtureHostFeature):
        contribution_prefix = "restart-conflict"

        @property
        def contribution_owner(self):
            return healthy.contribution_owner

    candidate = ConflictingRestartFeature()
    with pytest.raises(FeatureContributionCollectionError) as exc_info:
        await start_host_features([candidate], ctx)

    assert exc_info.value.feature is candidate
    assert isinstance(exc_info.value.__cause__, ContributionContractError)
    _assert_live(ctx, healthy, True)
    assert healthy.started
    assert not candidate.started
    await stop_host_features([healthy], ctx)


@pytest.mark.asyncio
async def test_host_collection_failure_is_sanitized_fatal_and_pre_mutation():
    original = RuntimeError("host-token=must-stay-private")

    class FailingHostFeature(SDKFixtureHostFeature):
        def get_setup_step_registrations(self):
            raise original

    feature = FailingHostFeature()
    ctx = SovereignHostContext()

    with pytest.raises(FeatureContributionCollectionError) as exc_info:
        await start_host_features([feature], ctx)

    error = exc_info.value
    assert error.feature is feature
    assert error.getter == "get_setup_step_registrations"
    assert "host-token" not in str(error)
    assert error.__cause__ is original
    assert not feature.started
    assert ctx.feature_contribution_runtime.active_owners() == ()


@pytest.mark.asyncio
async def test_host_stop_removes_fixture_routes_ui_and_declarations():
    feature = SDKFixtureHostFeature()
    ctx = SovereignHostContext()
    app = FastAPI()
    mount_host_feature_routers(app, [feature])
    mount_host_feature_ui(app, [feature])
    await start_host_features([feature], ctx)

    with TestClient(app) as client:
        assert client.get("/api/host-fixture/fixture").status_code == 200
        assert app.state.host_ui_manifest

        await stop_host_features([feature], ctx)
        unmount_host_features(app)
        assert client.get("/api/host-fixture/fixture").status_code == 404
        assert app.state.host_ui_manifest == []
        _assert_live(ctx, feature, False)


@pytest.mark.asyncio
async def test_a_later_start_does_not_erase_an_earlier_rejection():
    """Incremental starts are supported, so a second call must not blank the first.

    `start_host_features` may be called repeatedly against one context — that is
    what `previously_started` exists for. Assigning the rejection list made
    health stop reporting a still-refused feature the moment a later, unrelated
    feature started cleanly: the diagnostic vanishing on someone else's success
    (#2951).
    """
    import dataclasses

    from kestrel_sdk.signals import Trust

    ctx = SovereignHostContext()
    # An empty start primes the contribution runtime (and so `ctx.signal_registry`)
    # through the public path rather than reaching for the internals.
    await start_host_features([], ctx)

    refused = SDKFixtureHostFeature()
    other_trust = (
        Trust.UNTRUSTED if refused.source.trust is Trust.TRUSTED else Trust.TRUSTED
    )
    ctx.signal_registry.register(
        dataclasses.replace(refused.source, trust=other_trust)
    )

    await start_host_features([refused], ctx)
    after_first = tuple(ctx.rejected_host_feature_contributions)
    assert len(after_first) == 1                      # the premise: it WAS refused
    assert refused.source.name in after_first[0].reason

    class _PeerHostFeature(SDKFixtureHostFeature):
        contribution_prefix = "peer-host"

    # A LATER, unrelated start succeeds...
    peer = _PeerHostFeature()
    await start_host_features([peer], ctx)

    # ...and the earlier, still-unresolved rejection is still reported, intact.
    assert tuple(ctx.rejected_host_feature_contributions) == after_first
