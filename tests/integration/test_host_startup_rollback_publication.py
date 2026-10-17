"""Actual empty-host startup must not publish after unsafe contribution rollback.

This uses no agents/providers and claims no constitutional genesis or READY.
Only discovery selects a real SDK fixture input; runtime, SQLite, manager,
startup publication gate and teardown are the actual production components.
"""

import sqlite3

import pytest
from fastapi import FastAPI

from kestrel_sovereign.features.contribution_runtime import (
    FeatureContributionRuntimeError,
)
from tests.fixtures.sdk_contribution_fixture import SDKFixtureHostFeature
from tests.unit.test_execution_custody_review43_native import carries


@pytest.mark.asyncio
async def test_actual_host_startup_rejects_retained_declarative_failure(
    tmp_path, monkeypatch
):
    # Empty actual roster prevents ANY agent/provider creation or model request.
    config = tmp_path / "multi_agent.toml"
    config.write_text('[host]\nbind = "127.0.0.1"\nport = 8888\n', encoding="utf-8")
    host_db = tmp_path / "host.db"
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "KESTREL_HOME": str(tmp_path),
        "PROJECT_DIR": str(tmp_path),
        "KESTREL_DB_PATH": str(tmp_path / "private-state"),
        "KESTREL_HOST_DB_PATH": str(host_db),
        "KESTREL_MULTI_AGENT_CONFIG": str(config),
        "KESTREL_MULTI_AGENT": "1",
        "KESTREL_API_KEY": "inert-native-host-auth-not-provider-credential",
        "KESTREL_DB_BACKEND": "sqlite",
        "KESTREL_HOLD_BACKEND": "sqlite",
        "KESTREL_PHOENIX_ENABLED": "0",
        "KESTREL_TRACING_ENABLED": "0",
    }.items():
        monkeypatch.setenv(key, value)
    from kestrel_sovereign import host_features, server

    contexts, gate_states = [], []
    original = RuntimeError("ordinary original host feature start failure")

    class DriftDuringStart(SDKFixtureHostFeature):
        async def on_host_start(self, ctx):
            await super().on_host_start(ctx)
            contexts.append(ctx)
            await ctx.db.execute("CREATE TABLE actual_start_ack (id INTEGER)")
            await ctx.db.execute("INSERT INTO actual_start_ack VALUES (22)")
            ctx.wait_registry.deregister(
                self.wait_registration.name, self.wait_provider
            )
            assert not ctx.wait_registry.contains(
                self.wait_registration.name, self.wait_provider
            )
            raise original

    feature = DriftDuringStart()
    monkeypatch.setattr(
        host_features, "instantiate_host_features", lambda **kwargs: [feature]
    )
    app, failure = FastAPI(), None
    try:
        try:
            async with server._lifespan_startup(app):
                gate_states.append(app.state.host_context_publication_gate.is_set())
        except BaseException as error:
            failure = error
        assert len(contexts) == 1
        ctx = contexts[0]
        assert app.state.agent_manager is not None
        assert app.state.agent_manager.list_agents() == {}
        retained = [
            record.error
            for record in ctx._native_close_owners.values()
            if record.resource is feature
        ]
        assert retained and any(
            isinstance(error, FeatureContributionRuntimeError) for error in retained
        )
        with sqlite3.connect(host_db) as observer:
            assert observer.execute(
                "SELECT COUNT(*) FROM actual_start_ack WHERE id=22"
            ).fetchone() == (1,)
        assert failure is not None, (
            "server published an unsafe declarative rollback as an optional ordinary start failure"
        )
        assert failure is original and any(
            carries(failure, error) for error in retained
        )
        assert (
            not app.state.host_context_publication_gate.is_set() and gate_states == []
        )
        assert app.state.host_context is ctx
    finally:
        # The actual outer server drain retires every independent native owner.
        # A retained unsafe declarative receipt is expected, not a cleanup ACK.
        await server._shutdown_server_resources(app)
        for ctx in contexts:
            assert ctx.db.backend._connection is None
