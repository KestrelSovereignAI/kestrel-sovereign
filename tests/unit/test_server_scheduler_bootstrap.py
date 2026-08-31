"""Host scheduler protocol bootstrap sequencing regressions."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from kestrel_sovereign import server


@pytest.mark.asyncio
async def test_protocol_preflight_seeds_all_resolved_dids_without_polling(
    monkeypatch,
) -> None:
    """All local DIDs are established before any agent feature runner starts."""
    events: list[str] = []
    storage_instances = []
    runner_instances = []

    class _Storage:
        def __init__(self, *, backend, dsn) -> None:
            assert backend == "postgres"
            assert dsn == "postgresql://scheduler-test"
            self.db = object()
            self.closed = False
            storage_instances.append(self)

        async def initialize(self) -> None:
            events.append("storage-initialize")

        async def close(self) -> None:
            self.closed = True
            events.append("storage-close")

    class _Runner:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs
            self.started = False
            runner_instances.append(self)

        async def _ensure_tables(self) -> None:
            events.append("protocol-seeded")

        async def start(self) -> None:  # pragma: no cover - must not be called
            self.started = True
            raise AssertionError("protocol preflight must not start polling")

    agent_configs = {
        "did:pkh:configured-warm": ("Warm", object()),
        "did:pkh:configured-cold": ("Cold", object()),
    }

    async def _resolve(_config):
        events.append("read-only-did-discovery")
        return agent_configs

    manager = SimpleNamespace(
        local_agent_configs_by_did=AsyncMock(side_effect=_resolve),
        cold_scheduler_identity_failures=[],
        is_scheduler_agent_authorized=lambda _did: True,
        set_scheduler_polling_managed_by_host=MagicMock(),
    )
    app = FastAPI()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setattr(
        "kestrel_sovereign.storage.async_storage.AsyncStorage", _Storage
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner.SchedulerRunner", _Runner
    )

    await server._prepare_shared_postgres_scheduler_protocol(
        app, manager, object()
    )

    assert events == [
        "read-only-did-discovery",
        "storage-initialize",
        "protocol-seeded",
        "storage-close",
    ]
    assert tuple(runner_instances[0].kwargs["authorized_agent_ids"]) == tuple(
        agent_configs
    )
    assert runner_instances[0].started is False
    assert storage_instances[0].closed is True
    manager.set_scheduler_polling_managed_by_host.assert_called_once_with(True)


@pytest.mark.asyncio
async def test_protocol_preflight_keeps_healthy_dids_when_one_is_unresolved(
    monkeypatch,
) -> None:
    """An unavailable configured identity is latched but cannot block peers."""
    storage_instances = []
    runner_instances = []

    class _Storage:
        def __init__(self, **_kwargs) -> None:
            self.db = object()
            self.closed = False
            storage_instances.append(self)

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    class _Runner:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs
            runner_instances.append(self)

        async def _ensure_tables(self) -> None:
            return None

    missing_identity = RuntimeError("identity database is not initialized")
    manager = SimpleNamespace(
        local_agent_configs_by_did=AsyncMock(
            return_value={"did:pkh:healthy": ("Healthy", object())}
        ),
        cold_scheduler_identity_failures=[("Unincepted", missing_identity)],
    )
    app = FastAPI()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setattr(
        "kestrel_sovereign.storage.async_storage.AsyncStorage", _Storage
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner.SchedulerRunner", _Runner
    )

    await server._prepare_shared_postgres_scheduler_protocol(
        app, manager, object()
    )

    assert tuple(runner_instances[0].kwargs["authorized_agent_ids"]) == (
        "did:pkh:healthy",
    )
    assert storage_instances[0].closed is True
    assert app.state.scheduler_readiness_failures == [
        {
            "agent": "Unincepted",
            "scope": "identity",
            "state": "unavailable",
            "error_code": "scheduler_identity_unavailable",
            "cause_type": "RuntimeError",
        }
    ]


@pytest.mark.asyncio
async def test_protocol_preflight_cancellation_closes_its_temporary_storage(
    monkeypatch,
) -> None:
    """A cancelled schema-only bootstrap cannot strand an unseen DB pool."""
    entered_initialize = asyncio.Event()
    release_initialize = asyncio.Event()
    storage_instances = []

    class _BlockingStorage:
        def __init__(self, **_kwargs) -> None:
            self.db = None
            self.closed = False
            storage_instances.append(self)

        async def initialize(self) -> None:
            entered_initialize.set()
            await release_initialize.wait()

        async def close(self) -> None:
            self.closed = True

    manager = SimpleNamespace(
        local_agent_configs_by_did=AsyncMock(
            return_value={"did:pkh:healthy": ("Healthy", object())}
        ),
        cold_scheduler_identity_failures=[],
    )
    app = FastAPI()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setattr(
        "kestrel_sovereign.storage.async_storage.AsyncStorage", _BlockingStorage
    )

    bootstrap = asyncio.create_task(
        server._prepare_shared_postgres_scheduler_protocol(app, manager, object())
    )
    await entered_initialize.wait()
    bootstrap.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bootstrap

    assert storage_instances[0].closed is True


@pytest.mark.asyncio
async def test_lifespan_preflights_before_parallel_agent_initialization(
    monkeypatch,
    tmp_path,
) -> None:
    """Hold state exists before preflight, agent load, and the host runner."""
    from kestrel_sovereign import host_features as hf
    from kestrel_sovereign.a2a import did_registry
    from kestrel_sovereign.multi_agent import agent_manager, config as ma_config
    from kestrel_sovereign import phoenix_supervisor as phoenix_module
    from kestrel_sovereign.security import demo_isolation

    config_path = tmp_path / "multi_agent.toml"
    config_path.write_text("[host]\nport = 8888\n")
    fake_config = SimpleNamespace(
        host=SimpleNamespace(bind="127.0.0.1", port=8888), agents={}
    )
    events: list[str] = []
    hold_store = object()
    host_context = SimpleNamespace(
        hold_store=hold_store,
        hold_db=None,
        db=None,
        session_factory=None,
        feature_contribution_runtime=None,
    )

    class _Manager:
        init_failures = []

        def set_agent_registration_hook(self, _hook) -> None:
            return None

        async def load_from_config(self, config) -> int:
            assert config is fake_config
            events.append("load")
            return 0

        def list_agents(self):
            return {}

    manager = _Manager()

    async def _preflight(app, supplied_manager, config) -> None:
        assert supplied_manager is manager
        assert config is fake_config
        events.append("preflight")

    async def _start(app, supplied_manager, config) -> None:
        assert supplied_manager is manager
        assert config is fake_config
        events.append("host-start")

    async def _build_host_context(*, config):
        assert isinstance(config, dict)
        events.append("context-build")
        return host_context

    monkeypatch.setenv("KESTREL_MULTI_AGENT", "1")
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "0")
    monkeypatch.setattr(server, "resolve_multi_agent_path", lambda _env: config_path)
    monkeypatch.setattr(ma_config.MultiAgentConfig, "load", lambda *_a, **_k: fake_config)
    monkeypatch.setattr(agent_manager, "AgentManager", lambda **_k: manager)
    monkeypatch.setattr(
        server, "_prepare_shared_postgres_scheduler_protocol", _preflight
    )
    monkeypatch.setattr(server, "_start_host_scheduler", _start)
    monkeypatch.setattr(did_registry, "install_a2a_did_resolver", lambda *_a, **_k: None)
    monkeypatch.setattr(phoenix_module, "should_supervise_phoenix", lambda: False)
    monkeypatch.setattr(demo_isolation, "classify_server_mode", lambda _agents: False)
    monkeypatch.setattr(server, "_mount_feature_ui_assets", lambda _app: None)
    monkeypatch.setattr(server, "_mount_feature_routers", lambda _app: None)
    monkeypatch.setattr(server, "setup_tracing", lambda _app: None)
    monkeypatch.setattr(hf, "instantiate_host_features", lambda **_k: [])
    monkeypatch.setattr(hf, "build_host_context", _build_host_context)

    app = FastAPI()
    async with server._lifespan_startup(app):
        pass

    assert events == ["context-build", "preflight", "load", "host-start"]
    assert app.state.host_context is host_context
    assert app.state.host_context.hold_store is hold_store


@pytest.mark.asyncio
async def test_single_agent_identity_conflict_precedes_hold_custody_binding(
    monkeypatch,
    tmp_path,
) -> None:
    """A foreign runtime database is refused before Hold can bind its custody."""

    from kestrel_sovereign import host_features as hf
    from kestrel_sovereign import phoenix_supervisor as phoenix_module

    events: list[str] = []

    async def _reject_foreign_database(*_args, **_kwargs) -> str:
        events.append("identity-preflight")
        raise ValueError("Identity conflict: configured database belongs elsewhere")

    async def _build_control_context(_app, _config) -> object:
        events.append("hold-custody-binding")
        return object()

    missing_config = tmp_path / "missing-multi-agent.toml"
    monkeypatch.delenv("KESTREL_MULTI_AGENT", raising=False)
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://foreign/runtime")
    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path / "local-anchor"))
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "0")
    monkeypatch.setattr(
        server,
        "resolve_multi_agent_path",
        lambda _env: missing_config,
    )
    monkeypatch.setattr(server, "get_agent_did_async", _reject_foreign_database)
    monkeypatch.setattr(server, "_build_host_control_context", _build_control_context)
    monkeypatch.setattr(phoenix_module, "should_supervise_phoenix", lambda: False)
    monkeypatch.setattr(server, "_mount_feature_ui_assets", lambda _app: None)
    monkeypatch.setattr(server, "_mount_feature_routers", lambda _app: None)
    monkeypatch.setattr(server, "setup_tracing", lambda _app: None)
    monkeypatch.setattr(hf, "instantiate_host_features", lambda **_kwargs: [])

    app = FastAPI()
    async with server._lifespan_startup(app):
        pass

    assert events == ["identity-preflight"]
    assert "Identity conflict" in app.state.startup_error
    assert app.state.host_context is None
