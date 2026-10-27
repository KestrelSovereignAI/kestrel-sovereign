"""Actual empty-fleet server startup publishes only the original host registry.

No agent inference/genesis/READY claim; native host SQLite, registries, manager,
publication gate and complete shutdown are production components.
"""

import pytest
from fastapi import FastAPI
from kestrel_sdk.operator import (
    OperatorAuthorizationError,
    ServiceRequirement,
    ServiceScope,
)
from tests.fixtures.sdk_contribution_fixture import SDKFixtureHostFeature


def _configure_empty_fleet(tmp_path, monkeypatch):
    config = tmp_path / "multi_agent.toml"
    config.write_text('[host]\nbind="127.0.0.1"\nport=8888\n')
    for key, value in {
        "KESTREL_HOME": str(tmp_path),
        "PROJECT_DIR": str(tmp_path),
        "KESTREL_DB_PATH": str(tmp_path / "private"),
        "KESTREL_HOST_DB_PATH": str(tmp_path / "host.db"),
        "KESTREL_MULTI_AGENT_CONFIG": str(config),
        "KESTREL_MULTI_AGENT": "1",
        "KESTREL_API_KEY": "host-service-native-startup-test-only",
        "KESTREL_DB_BACKEND": "sqlite",
        "KESTREL_HOLD_BACKEND": "sqlite",
        "KESTREL_PHOENIX_ENABLED": "0",
        "KESTREL_TRACING_ENABLED": "0",
    }.items():
        monkeypatch.setenv(key, value)


async def test_actual_server_startup_and_retirement_publish_exact_host_services(
    tmp_path, monkeypatch
):
    _configure_empty_fleet(tmp_path, monkeypatch)
    from kestrel_sovereign import host_features, server

    feature = SDKFixtureHostFeature()
    monkeypatch.setattr(
        host_features, "instantiate_host_features", lambda **kwargs: [feature]
    )
    app = FastAPI()
    try:
        async with server._lifespan_startup(app):
            manager, ctx = app.state.agent_manager, app.state.host_context
            assert manager.list_agents() == {}
            assert app.state.host_context_publication_gate.is_set()
            assert manager._host_service_resolver is ctx.operator_registry
            assert (
                manager._host_context_publication_state.service_resolver
                is ctx.operator_registry
            )
            requirement = ServiceRequirement(
                feature.service_registration.descriptor.name, "1.0.0", ServiceScope.HOST
            )
            async with ctx.operator_registry.host_service_operation(
                requirement
            ) as original:
                assert original.service is feature.service
            with pytest.raises(OperatorAuthorizationError):
                original.check_current()
    finally:
        await server._shutdown_server_resources(app)
    assert manager._host_service_resolver is None
    assert ctx.db.backend._connection is None


async def test_retired_retained_agent_refuses_before_host_routes_are_published(
    tmp_path, monkeypatch
):
    """A real fleet binding refusal must not advertise the stopped candidate."""
    _configure_empty_fleet(tmp_path, monkeypatch)
    from kestrel_sovereign import host_features, server
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    app = FastAPI()
    healthy = KestrelAgent(
        did="did:test:preflight-healthy",
        storage_path=str(tmp_path / "healthy.db"),
        sync_enabled=False,
    )
    retained = KestrelAgent(
        did="did:test:preflight-retained",
        storage_path=str(tmp_path / "retained.db"),
        sync_enabled=False,
    )

    class RetainedAgentHostFeature(SDKFixtureHostFeature):
        async def on_host_start(self, ctx):
            await super().on_host_start(ctx)
            manager = app.state.agent_manager
            manager._register_agent("Healthy", healthy)
            manager._register_agent("Retained", retained)
            # Inject only the retained lifecycle state from the shutdown prefix.
            # Native manager, bindings, host startup, registries and cleanup run.
            retained._host_services.retire()

    feature = RetainedAgentHostFeature()
    monkeypatch.setattr(
        host_features, "instantiate_host_features", lambda **kwargs: [feature]
    )
    try:
        async with server._lifespan_startup(app):
            ctx = app.state.host_context
            manager = app.state.agent_manager
            assert feature.started and feature.stopped
            assert app.state.host_features == []
            assert not any(
                getattr(route, "path", None) == "/api/host-fixture/fixture"
                for route in app.routes
            )
            assert manager._host_service_resolver is None
            assert healthy._host_services._resolver is None
            assert retained._host_services._retired
    finally:
        ctx = getattr(app.state, "host_context", None)
        await server._shutdown_server_resources(app)
        if ctx is not None:
            assert ctx.db.backend._connection is None
