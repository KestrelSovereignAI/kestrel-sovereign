"""Actual single-agent server startup, not manually supplied runtime fences."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ProcessRuntimeExecutionFence,
    current_execution_custody,
)


def stub_single_startup(monkeypatch, tmp_path, *, backend="postgres"):
    from kestrel_sovereign import server, host_features
    from kestrel_sovereign.security import demo_isolation

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KESTREL_MULTI_AGENT", raising=False)
    monkeypatch.setenv("KESTREL_DB_BACKEND", backend)
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://provider-free-startup")
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "0")
    monkeypatch.setattr(server, "resolve_multi_agent_path", lambda _env: tmp_path / "absent.toml")
    monkeypatch.setattr(server, "get_agent_did_async", AsyncMock(return_value="did:test:single-startup"))
    monkeypatch.setattr(server, "verify_identity_isolation", lambda _did: None)
    monkeypatch.setattr(server, "_initialize_stop_receipts", AsyncMock())
    context = SimpleNamespace(hold_store=object())

    async def host_context(app, _config):
        app.state.host_context = context
        return context

    monkeypatch.setattr(server, "_build_host_control_context", host_context)
    monkeypatch.setattr(server, "_attach_stop_runtime", lambda *args: None)
    monkeypatch.setattr(server, "_mount_feature_ui_assets", lambda _app: None)
    monkeypatch.setattr(server, "_mount_feature_routers", lambda _app: None)
    monkeypatch.setattr(server, "_complete_deferred_agent_readiness", AsyncMock())
    monkeypatch.setattr(server, "setup_tracing", lambda _app: None)
    monkeypatch.setattr(host_features, "instantiate_host_features", lambda **kwargs: [])
    monkeypatch.setattr(demo_isolation, "classify_server_mode", lambda _agents: False)
    return server, FastAPI()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["none", "provider", "constructor", "initialize", "cancel"])
async def test_actual_single_pg_startup_owns_custody_before_provider_birth(monkeypatch, tmp_path, failure):
    server, app = stub_single_startup(monkeypatch, tmp_path)
    scopes, close_denials, candidates = [], [], []
    error = asyncio.CancelledError("failed birth") if failure == "cancel" else ValueError("failed birth")

    async def close():
        with pytest.raises(ExecutionAuthorityError, match="retired"):
            scopes[0].require_work()
        close_denials.append(True)

    service = SimpleNamespace(close=close)

    def provider():
        captured = current_execution_custody()
        assert len(captured) == 1, "single-agent provider born without original custody"
        scopes.append(captured[0])
        assert isinstance(scopes[0].fence, ProcessRuntimeExecutionFence)
        if failure == "provider":
            raise error
        return service

    def construct(**kwargs):
        assert kwargs["execution_custody"] is scopes[0]
        assert kwargs["llm_service"] is service
        if failure == "constructor":
            raise error

        async def initialize():
            if failure in {"initialize", "cancel"}:
                raise error

        agent = SimpleNamespace(
            did=kwargs["did"], _execution_custody=kwargs["execution_custody"],
            defer_agent_readiness_to_host=lambda: None, initialize=initialize,
            shutdown=close,
        )
        candidates.append(agent)
        return agent

    monkeypatch.setattr(server, "LLMService", provider)
    monkeypatch.setattr(server, "KestrelAgent", construct)
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError) as caught:
            async with server._lifespan_startup(app):
                pytest.fail("cancelled initialization published an agent")
        assert caught.value is error
    else:
        async with server._lifespan_startup(app):
            pass
    assert len(scopes) == 1
    if failure == "none":
        scopes[0].require_work()
        assert app.state.agent is candidates[0]
        assert not close_denials
    else:
        with pytest.raises(ExecutionAuthorityError, match="retired"):
            scopes[0].require_work()
        if failure in {"initialize", "cancel"}:
            assert not close_denials
            assert app.state.startup_cleanup_agent is candidates[0]
            await server._shutdown_server_agents(app)
            assert app.state.startup_cleanup_agent is None
        assert close_denials == ([] if failure == "provider" else [True])
        assert app.state.agent is None


@pytest.mark.asyncio
async def test_actual_single_sqlite_startup_keeps_unbound_compatibility(monkeypatch, tmp_path):
    server, app = stub_single_startup(monkeypatch, tmp_path, backend="sqlite")
    seen = []

    def provider():
        assert not current_execution_custody()
        return object()

    def construct(**kwargs):
        assert kwargs.get("execution_custody") is None
        seen.append(kwargs)
        return SimpleNamespace(did=kwargs["did"], defer_agent_readiness_to_host=lambda: None, initialize=AsyncMock())

    monkeypatch.setattr(server, "LLMService", provider)
    monkeypatch.setattr(server, "KestrelAgent", construct)
    async with server._lifespan_startup(app):
        pass
    assert len(seen) == 1
