"""A co-hosted fleet without KESTREL_BASE_URL is named at startup (#3872)."""

import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from kestrel_sovereign import server
from kestrel_sovereign.multi_agent.config import (
    HostConfig,
    LocalAgentConfig,
    MultiAgentConfig,
)

WARNING = "KESTREL_BASE_URL is unset"


def _unpublished_warnings(records):
    return [
        record
        for record in records
        if record.levelno == logging.WARNING and WARNING in record.getMessage()
    ]


async def _start_host(monkeypatch, tmp_path, caplog, *, agents):
    """Run the real multi-agent lifespan to its yield over a stub manager."""
    from kestrel_sovereign import host_features as hf
    from kestrel_sovereign import phoenix_supervisor as phoenix_module
    from kestrel_sovereign.multi_agent import agent_manager
    from kestrel_sovereign.security import demo_isolation

    config = MultiAgentConfig(
        host=HostConfig(port=8877),
        agents={
            f"Agent{index}": LocalAgentConfig(
                data_dir=tmp_path / f"agent{index}", port=8801 + index
            )
            for index in range(agents)
        },
    )
    seen_before_load = []

    class _Manager:
        init_failures = []

        def set_agent_registration_hook(self, _hook):
            return None

        def reconcile_spawn_authority_restart_roster(self, supplied):
            return supplied

        def bind_hold_store(self, _store):
            return None

        async def load_from_config(self, supplied, *, restart_roster_reconciled):
            assert supplied is config and restart_roster_reconciled is True
            # No agent has initialized, so none can have attempted a send.
            seen_before_load.extend(_unpublished_warnings(caplog.records))
            return agents

        def list_agents(self):
            return {}

    async def _host_context(app, _config):
        app.state.host_context = SimpleNamespace(
            hold_store=object(), feature_contribution_runtime=None
        )
        return app.state.host_context

    async def _noop(*_args):
        return None

    monkeypatch.setenv("KESTREL_MULTI_AGENT", "1")
    monkeypatch.setenv("KESTREL_API_KEY", "cohost-startup-test-key")
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "0")
    for name in ("PORT", "KESTREL_SERVER_HOST", "KESTREL_DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        server, "resolve_multi_agent_path", lambda _env: tmp_path / "absent.toml"
    )
    monkeypatch.setattr(MultiAgentConfig, "load", lambda *_a, **_k: config)
    monkeypatch.setattr(agent_manager, "AgentManager", lambda **_k: _Manager())
    monkeypatch.setattr(server, "_build_host_control_context", _host_context)
    monkeypatch.setattr(server, "_initialize_stop_receipts", _noop)
    monkeypatch.setattr(server, "_prepare_shared_postgres_scheduler_protocol", _noop)
    monkeypatch.setattr(server, "_start_host_scheduler", _noop)
    monkeypatch.setattr(phoenix_module, "should_supervise_phoenix", lambda: False)
    monkeypatch.setattr(demo_isolation, "classify_server_mode", lambda _agents: False)
    monkeypatch.setattr(server, "_mount_feature_ui_assets", lambda _app: None)
    monkeypatch.setattr(server, "_mount_feature_routers", lambda _app: None)
    monkeypatch.setattr(server, "setup_tracing", lambda _app: None)
    monkeypatch.setattr(hf, "instantiate_host_features", lambda **_k: [])
    caplog.set_level(logging.WARNING, logger=server.logger.name)

    app = FastAPI()
    async with server._lifespan_startup(app):
        assert app.state.startup_error is None
    return seen_before_load


async def test_two_agent_host_without_base_url_warns_once_before_agents_load(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.delenv("KESTREL_BASE_URL", raising=False)
    seen_before_load = await _start_host(monkeypatch, tmp_path, caplog, agents=2)
    [warning] = _unpublished_warnings(caplog.records)
    assert seen_before_load == [warning]
    message = warning.getMessage()
    assert "none of this host's 2 agents publishes a native A2A interface" in message
    assert "KESTREL_BASE_URL=http://127.0.0.1:8877" in message
    assert "KESTREL_A2A_PUSH_ENABLED=false" in message


@pytest.mark.parametrize(
    ("agents", "base_url"),
    [(1, None), (2, "http://127.0.0.1:8877")],
    ids=["single-agent", "base-url-set"],
)
async def test_host_that_can_publish_or_has_no_peer_does_not_warn(
    monkeypatch, tmp_path, caplog, agents, base_url
):
    if base_url is None:
        monkeypatch.delenv("KESTREL_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("KESTREL_BASE_URL", base_url)
    await _start_host(monkeypatch, tmp_path, caplog, agents=agents)
    assert _unpublished_warnings(caplog.records) == []
