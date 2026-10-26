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


async def test_actual_server_startup_and_retirement_publish_exact_host_services(
    tmp_path, monkeypatch
):
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
