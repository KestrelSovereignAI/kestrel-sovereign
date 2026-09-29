from __future__ import annotations

import asyncio
import traceback
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kestrel_sdk.features import ContextClauseRegistration, ContributionContractError

from kestrel_sovereign.agent.context_builder import ContextBuilder
from kestrel_sovereign.features.contribution_runtime import (
    FeatureContributionCollectionError,
    FeatureContributionRuntimeError,
)
from kestrel_sovereign.host_features import (
    SovereignHostContext,
    mount_host_feature_routers,
    mount_host_feature_ui,
    start_host_features,
    stop_host_features,
    unmount_host_features,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from tests.fixtures.sdk_contribution_fixture import (
    SDKFixtureFeature,
    SDKFixtureHostFeature,
)


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
async def test_host_context_clauses_reach_existing_agent_prompts_and_teardown():
    ctx = SovereignHostContext()
    feature = SDKFixtureHostFeature()
    await start_host_features([feature], ctx)
    agent = KestrelAgent(did="did:test:host-context-clause", sync_enabled=False)
    agent._ensure_feature_contribution_runtime()
    agent.context_builder = ContextBuilder(
        MagicMock(),
        context_clause_registry=agent.context_clause_registry,
    )

    agent.bind_host_context_clause_registry(
        ctx.feature_contribution_runtime.context_clause_registry
    )

    prompt = agent.context_builder.build_system_prompt(
        "C", include_briefing=False
    )
    assert "stable context from host-fixture" in prompt

    await stop_host_features([feature], ctx)
    prompt_after_stop = agent.context_builder.build_system_prompt(
        "C", include_briefing=False
    )
    assert "stable context from host-fixture" not in prompt_after_stop


@pytest.mark.asyncio
async def test_later_host_context_collision_is_rejected_before_rendering(tmp_path):
    """A host registry must retain reverse visibility into bound agents."""

    ctx = SovereignHostContext()
    await start_host_features([], ctx)
    agent = KestrelAgent(
        did="did:test:late-host-context-collision",
        storage_path=str(tmp_path / "agent.db"),
        sync_enabled=False,
    )
    agent_feature = SDKFixtureFeature(agent)
    agent_runtime = agent._ensure_feature_contribution_runtime()
    agent_runtime.activate(
        agent_runtime.prepare_transition((agent_feature,)).only()
    )
    agent.bind_host_context_clause_registry(
        ctx.feature_contribution_runtime.context_clause_registry
    )

    class LateHostFeature(SDKFixtureHostFeature):
        name = "late-host-fixture"
        contribution_prefix = "late-host-fixture"

    candidate = LateHostFeature()
    candidate.context_registration = ContextClauseRegistration(
        owner=candidate.contribution_owner,
        name=agent_feature.context_registration.name,
        priority=30,
        renderer=candidate._render_context_clause,
    )

    started = await start_host_features([candidate], ctx)

    assert started == []
    assert len(ctx.rejected_host_feature_contributions) == 1
    assert "context clause already registered" in (
        ctx.rejected_host_feature_contributions[0].reason
    )
    assert candidate.context_renderer_calls == 0
    assert ctx.feature_contribution_runtime.active_owners() == ()


@pytest.mark.asyncio
async def test_later_host_context_respects_bound_agent_bootstrap_names(tmp_path):
    """An agent's custom bootstrap namespace is visible to later host starts."""

    ctx = SovereignHostContext()
    await start_host_features([], ctx)
    agent = KestrelAgent(
        did="did:test:host-bootstrap-collision",
        storage_path=str(tmp_path / "agent.db"),
        sync_enabled=False,
    )
    runtime = agent._ensure_feature_contribution_runtime()
    agent.context_builder = ContextBuilder(
        MagicMock(),
        agent_data_path=str(tmp_path),
        context_clause_registry=runtime.context_clause_registry,
    )
    assert agent.context_builder._bootstrap_loader.add_file("POLICY.yaml")
    agent.bind_host_context_clause_registry(
        ctx.feature_contribution_runtime.context_clause_registry
    )

    class LateHostFeature(SDKFixtureHostFeature):
        name = "late-bootstrap-host-fixture"
        contribution_prefix = "late-bootstrap-host-fixture"

    candidate = LateHostFeature()
    candidate.context_registration = ContextClauseRegistration(
        owner=candidate.contribution_owner,
        name="POLICY.yaml",
        priority=30,
        renderer=candidate._render_context_clause,
    )

    with pytest.raises(
        FeatureContributionRuntimeError,
        match="reserved host audit name",
    ):
        await start_host_features([candidate], ctx)

    assert candidate.context_renderer_calls == 0
    assert ctx.feature_contribution_runtime.active_owners() == ()


@pytest.mark.asyncio
async def test_host_renderer_failure_does_not_leak_feature_repr_or_cause():
    """The sanitized collection boundary must cross host activation intact."""

    secret = "api-key=host-renderer-must-stay-private"
    original = RuntimeError(secret)

    class SecretReprHostFeature(SDKFixtureHostFeature):
        name = "secret-repr-host-fixture"
        contribution_prefix = "secret-repr-host-fixture"

        def __repr__(self):
            return f"<SecretReprHostFeature {secret}>"

    feature = SecretReprHostFeature()

    def fail_renderer():
        raise original

    object.__setattr__(feature.context_registration, "renderer", fail_renderer)
    ctx = SovereignHostContext()

    with pytest.raises(FeatureContributionCollectionError) as exc_info:
        await start_host_features([feature], ctx)

    error = exc_info.value
    assert error.feature is feature
    assert error.getter == "render_context_clauses"
    assert secret not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    assert ctx.feature_contribution_runtime.active_owners() == ()
    assert not feature.started


@pytest.mark.asyncio
async def test_host_renderer_cancelled_error_is_sanitized_without_cause():
    """A synchronous renderer cannot forge task cancellation with secret text."""

    secret = "api-key=renderer-cancel-must-stay-private"
    feature = SDKFixtureHostFeature()

    def cancel_renderer():
        raise asyncio.CancelledError(secret)

    object.__setattr__(feature.context_registration, "renderer", cancel_renderer)
    ctx = SovereignHostContext()

    with pytest.raises(FeatureContributionCollectionError) as exc_info:
        await start_host_features([feature], ctx)

    error = exc_info.value
    assert error.feature is feature
    assert error.getter == "render_context_clauses"
    assert secret not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    assert ctx.feature_contribution_runtime.active_owners() == ()
    assert not feature.started


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
    secret = "api-key=host-activation-must-stay-private"

    class FirstHostFeature(SDKFixtureHostFeature):
        name = "first-host-fixture"
        contribution_prefix = "first-host-fixture"

    class SecondHostFeature(SDKFixtureHostFeature):
        name = "second-host-fixture"
        contribution_prefix = "second-host-fixture"

        def __repr__(self):
            return f"<SecondHostFeature {secret}>"

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
    ) as exc_info:
        await start_host_features([first, second], ctx)

    assert secret not in str(exc_info.value)
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
        # A genuinely DIFFERENT feature: overriding the contribution prefix
        # alone leaves `.name` shared, which the supersede-by-name rule
        # correctly reads as a retry of the same feature.
        contribution_prefix = "peer-host"
        name = "peer-host"

    # A LATER, unrelated start succeeds...
    peer = _PeerHostFeature()
    await start_host_features([peer], ctx)

    # ...and the earlier, still-unresolved rejection is still reported, intact.
    assert tuple(ctx.rejected_host_feature_contributions) == after_first


# ---------------------------------------------------------------------------
# #3058 — a dropped host feature must not read as a healthy one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failed_start_hook_is_recorded_where_health_reads_it():
    """A start-hook failure drops the feature WHOLE — router, panel, service.

    Every consumer downstream then sees an empty list rather than an error.
    #3058: WorkflowsHostFeature refused a host with no database, the feature
    vanished, Talon's router refused without it, and /health reported ok over
    an entire missing operator run plane — the only trace a single WARNING.

    "Did not start" and "was refused activation" are the same fact for a
    reader: the feature is not loaded. So it goes through the same door.
    """
    class FailingHostFeature(SDKFixtureHostFeature):
        contribution_prefix = "unstarted-host-fixture"
        name = "unstarted-host-fixture"

        async def on_host_start(self, ctx):
            raise RuntimeError("operator runs require the host database")

    ctx = SovereignHostContext()

    started = await start_host_features([FailingHostFeature()], ctx)

    assert started == []
    rejections = tuple(ctx.rejected_host_feature_contributions)
    assert [r.feature_name for r in rejections] == ["unstarted-host-fixture"]
    assert "operator runs require the host database" in rejections[0].reason


@pytest.mark.asyncio
async def test_a_feature_that_starts_on_retry_stops_being_reported():
    """The verdict is superseded by NAME, like every other rejection.

    A fixed feature is retried as a freshly constructed instance, so an
    id-keyed match would never fire and health would go on reporting a
    now-live feature as not loaded.
    """
    class Flaky(SDKFixtureHostFeature):
        contribution_prefix = "flaky-host-fixture"
        name = "flaky-host-fixture"
        fail = True

        async def on_host_start(self, ctx):
            if type(self).fail:
                raise RuntimeError("not yet")
            await super().on_host_start(ctx)

    ctx = SovereignHostContext()
    await start_host_features([Flaky()], ctx)
    assert [r.feature_name for r in ctx.rejected_host_feature_contributions] == [
        "flaky-host-fixture"
    ]

    Flaky.fail = False
    try:
        started = await start_host_features([Flaky()], ctx)
    finally:
        Flaky.fail = True

    assert len(started) == 1
    assert tuple(ctx.rejected_host_feature_contributions) == ()


def test_a_host_with_no_backend_names_why_in_detailed_health():
    """The symptom without the cause is a diagnosis one scroll away.

    build_host_context is built to survive a store it cannot open — the host
    must start without one — but a capability gap has to be named. Without
    this the reason lives only in a boot log, while the features it took down
    each report an empty result.
    """
    from kestrel_sovereign.server import _with_host_feature_rejections

    class _State:
        host_context = SovereignHostContext(
            backend_error="TransactionError: no such column: slot"
        )

    payload = _with_host_feature_rejections(_State(), {"status": "healthy"})

    assert payload["host_backend_unavailable"] == (
        "TransactionError: no such column: slot"
    )
    assert payload["status"] == "degraded", (
        "a host running without its store is not healthy"
    )


def test_a_healthy_host_gains_no_diagnostic_noise():
    """The fold must stay invisible when there is nothing to report."""
    from kestrel_sovereign.server import _with_host_feature_rejections

    class _State:
        host_context = SovereignHostContext()

    payload = _with_host_feature_rejections(_State(), {"status": "healthy"})

    assert payload == {"status": "healthy"}


@pytest.mark.asyncio
async def test_build_host_context_carries_why_the_backend_is_missing(tmp_path):
    """`db=None` is the symptom; the exception that caused it is the fact.

    The host is built to start without a store, so this path must not raise —
    but reducing the cause to a log line is what made #3058 a six-link chain
    nobody could see the start of.
    """
    from kestrel_sovereign.host_features.context import build_host_context

    # A directory where the database file must be: the open fails, and it
    # fails for a reason that is not this test's invention.
    occupied = tmp_path / "host.db"
    occupied.mkdir()

    ctx = await build_host_context(config={}, db_path=str(occupied))

    assert ctx.db is None
    assert ctx.backend_error, "the reason must travel with the degradation"


@pytest.mark.asyncio
async def test_a_host_that_opened_its_backend_reports_no_error(tmp_path):
    """And the field stays empty when there is nothing wrong."""
    from kestrel_sovereign.host_features.context import build_host_context

    ctx = await build_host_context(config={}, db_path=str(tmp_path / "host.db"))
    try:
        assert ctx.db is not None
        assert ctx.backend_error == ""
    finally:
        if ctx.db is not None:
            await ctx.db.close()


def test_the_boolean_and_the_string_cannot_disagree():
    """`overall_healthy` is what monitors read; `status` is what humans read.

    Downgrading one and leaving the other is the silent-healthy report this
    diagnostic exists to end, arriving through whichever field the consumer
    happened to pick.
    """
    from kestrel_sovereign.server import _with_host_feature_rejections

    class _State:
        host_context = SovereignHostContext(backend_error="boom")

    payload = _with_host_feature_rejections(
        _State(), {"status": "healthy", "overall_healthy": True}
    )

    assert payload["status"] == "degraded"
    assert payload["overall_healthy"] is False


def test_a_healthy_payload_keeps_its_boolean():
    """And the fold does not invent a downgrade where there is none."""
    from kestrel_sovereign.server import _with_host_feature_rejections

    class _State:
        host_context = SovereignHostContext()

    payload = _with_host_feature_rejections(
        _State(), {"status": "healthy", "overall_healthy": True}
    )

    assert payload["overall_healthy"] is True


def test_a_non_string_backend_error_is_not_an_outage():
    """`str()` of any object is truthy, so coercion would invent a failure.

    Found by the suite, not by this file: `app.state` is a module singleton
    and an earlier test leaves a stand-in host context on it, whose every
    attribute answers. Coercing that attribute reported a host backend outage
    naming the stand-in's repr — a surface built to stop health lying, lying.
    """
    from unittest.mock import MagicMock

    from kestrel_sovereign.server import _with_host_feature_rejections

    class _State:
        host_context = MagicMock()

    payload = _with_host_feature_rejections(
        _State(), {"status": "healthy", "overall_healthy": True}
    )

    assert payload == {"status": "healthy", "overall_healthy": True}


def test_a_whitespace_backend_error_is_not_an_outage():
    """Nor is a field that exists and says nothing."""
    from kestrel_sovereign.server import _with_host_feature_rejections

    class _State:
        host_context = SovereignHostContext(backend_error="   ")

    payload = _with_host_feature_rejections(_State(), {"status": "healthy"})

    assert payload == {"status": "healthy"}
