"""A co-hosted fleet without KESTREL_BASE_URL is named at startup (#3872)."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from kestrel_sovereign import server
from kestrel_sovereign.multi_agent.config import (
    HostConfig,
    LocalAgentConfig,
    MultiAgentConfig,
)

WARNING = "KESTREL_BASE_URL is unset"
CHECK_FAILED = "Could not check whether this host publishes native A2A"


def _unpublished_warnings(records):
    return [
        record
        for record in records
        if record.levelno == logging.WARNING and WARNING in record.getMessage()
    ]


def _initialized_agent(name):
    agent = MagicMock()
    agent.agent_id = f"did:test:{name.lower()}"
    agent.initialize = AsyncMock()
    agent.shutdown = AsyncMock()

    async def no_durable_edges(_node_id):
        return []

    agent.storage = SimpleNamespace(get_edges_from=no_durable_edges)
    return agent


async def _start_host(monkeypatch, tmp_path, caplog, *, active):
    """Run the real multi-agent lifespan to its yield.

    Three agents are registered and the first ``active`` have
    ``autostart=true``. The real ``AgentManager.load_from_config`` selects
    which of them start, so the cold rows are left out exactly as on a real
    host; only agent initialization itself is replaced. Returns the names it
    initialized and the warnings already logged when the first one began.
    """
    from kestrel_sovereign import host_features as hf
    from kestrel_sovereign import phoenix_supervisor as phoenix_module
    from kestrel_sovereign.multi_agent import agent_manager
    from kestrel_sovereign.security import demo_isolation

    config = MultiAgentConfig(
        host=HostConfig(port=8877),
        agents={
            f"Agent{index}": LocalAgentConfig(
                data_dir=tmp_path / f"agent{index}",
                port=8801 + index,
                autostart=index < active,
            )
            for index in range(3)
        },
    )
    initialized = []
    logged_before_first_initialize = []

    async def initialize(name, _config):
        if not initialized:
            logged_before_first_initialize.extend(
                _unpublished_warnings(caplog.records)
            )
        initialized.append(name)
        return _initialized_agent(name)

    loader = agent_manager.AgentManager(base_data_dir=tmp_path)
    loader._initialize_agent = AsyncMock(side_effect=initialize)
    loader._on_agent_registered = AsyncMock()
    loader._run_hosted_agent_ready_hooks = AsyncMock()

    class _Manager:
        """The lifespan's manager; loading runs on the real ``loader``."""

        init_failures = []

        def set_agent_registration_hook(self, _hook):
            return None

        def reconcile_spawn_authority_restart_roster(self, supplied):
            return supplied

        def bind_hold_store(self, _store):
            return None

        async def load_from_config(
            self, supplied, *, restart_roster_reconciled, before_initialize
        ):
            assert supplied is config and restart_roster_reconciled is True
            return await loader.load_from_config(
                supplied,
                restart_roster_reconciled=True,
                before_initialize=before_initialize,
            )

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
    return initialized, logged_before_first_initialize


async def test_two_agent_host_without_base_url_warns_once_before_agents_load(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.delenv("KESTREL_BASE_URL", raising=False)
    initialized, logged_before_first_initialize = await _start_host(
        monkeypatch, tmp_path, caplog, active=2
    )
    assert initialized == ["Agent0", "Agent1"]
    [warning] = _unpublished_warnings(caplog.records)
    # No agent had initialized, so none could have attempted a send.
    assert logged_before_first_initialize == [warning]
    message = warning.getMessage()
    assert "none of this host's 2 agents publishes a native A2A interface" in message
    assert "KESTREL_BASE_URL=http://127.0.0.1:8877" in message
    assert "KESTREL_A2A_PUSH_ENABLED=false" in message


@pytest.mark.parametrize(
    ("active", "base_url"),
    [(1, None), (2, "http://127.0.0.1:8877")],
    ids=["single-agent-with-cold-registrations", "base-url-set"],
)
async def test_host_that_can_publish_or_has_no_peer_does_not_warn(
    monkeypatch, tmp_path, caplog, active, base_url
):
    if base_url is None:
        monkeypatch.delenv("KESTREL_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("KESTREL_BASE_URL", base_url)
    initialized, _logged = await _start_host(
        monkeypatch, tmp_path, caplog, active=active
    )
    # Three registrations, but the cold ones never start.
    assert len(initialized) == active
    assert _unpublished_warnings(caplog.records) == []


async def test_a_failing_publication_check_never_stops_the_host(
    monkeypatch, tmp_path, caplog
):
    from kestrel_sovereign.a2a import native_publication

    def broken(*_args, **_kwargs):
        raise RuntimeError("publication check broke")

    monkeypatch.delenv("KESTREL_BASE_URL", raising=False)
    monkeypatch.setattr(native_publication, "unpublished_cohost_warning", broken)
    initialized, _logged = await _start_host(
        monkeypatch, tmp_path, caplog, active=2
    )
    assert initialized == ["Agent0", "Agent1"]
    [failure] = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and CHECK_FAILED in record.getMessage()
    ]
    assert failure.exc_info[1].args == ("publication check broke",)
