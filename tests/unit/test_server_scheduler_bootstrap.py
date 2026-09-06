"""Host scheduler protocol bootstrap sequencing regressions."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from kestrel_sovereign import server


@pytest.mark.asyncio
async def test_server_owns_one_independently_capacity_sized_backend_for_all_agents(
    monkeypatch,
) -> None:
    instances = []

    class _Backend:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.connect = AsyncMock()
            self.close = AsyncMock()
            instances.append(self)

    app = FastAPI()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setenv("KESTREL_SHARED_AGENT_POSTGRES_MAX_POOL_SIZE", "31")
    monkeypatch.setenv(
        "KESTREL_SHARED_AGENT_POSTGRES_ADVISORY_MAX_POOL_SIZE", "5"
    )
    monkeypatch.setattr(
        "kestrel_sovereign.storage.db.postgres.PostgresBackend", _Backend
    )
    backend = await server._start_shared_agent_postgres_backend(app)

    assert backend is instances[0]
    assert backend.kwargs == {
        "dsn": "postgresql://scheduler-test",
        "min_pool_size": 2,
        "max_pool_size": 31,
        "advisory_max_pool_size": 5,
    }
    backend.connect.assert_awaited_once()
    app.state.agent_manager = None
    app.state.startup_cleanup_agent_manager = None
    await server._shutdown_shared_agent_postgres_backend(app)
    backend.close.assert_awaited_once()
    assert app.state.shared_agent_postgres_backend is None


@pytest.mark.asyncio
async def test_shared_agent_pool_defaults_are_not_scheduler_capacity(
    monkeypatch,
) -> None:
    instances = []

    class _Backend:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.connect = AsyncMock()
            instances.append(self)

    app = FastAPI()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setattr(
        "kestrel_sovereign.storage.db.postgres.PostgresBackend", _Backend
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.feature."
        "SchedulerFeature._load_max_concurrent_tasks",
        lambda: 1,
    )

    await server._start_shared_agent_postgres_backend(app)

    assert instances[0].kwargs == {
        "dsn": "postgresql://scheduler-test",
        "min_pool_size": 2,
        "max_pool_size": 20,
        "advisory_max_pool_size": 4,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,value",
    [
        ("KESTREL_SHARED_AGENT_POSTGRES_MAX_POOL_SIZE", "0"),
        ("KESTREL_SHARED_AGENT_POSTGRES_ADVISORY_MAX_POOL_SIZE", "many"),
    ],
)
async def test_shared_agent_pool_rejects_invalid_capacity(
    monkeypatch, name: str, value: str
) -> None:
    app = FastAPI()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        await server._start_shared_agent_postgres_backend(app)

    assert app.state.shared_agent_postgres_backend is None


@pytest.mark.asyncio
async def test_server_refuses_to_close_shared_pool_while_manager_is_live() -> None:
    app = FastAPI()
    backend = SimpleNamespace(close=AsyncMock())
    app.state.shared_agent_postgres_backend = backend
    app.state.agent_manager = object()
    app.state.startup_cleanup_agent_manager = None

    with pytest.raises(RuntimeError, match="manager still owns children"):
        await server._shutdown_shared_agent_postgres_backend(app)

    backend.close.assert_not_awaited()


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
    """The full multi-agent startup order is preflight → load → host runner."""
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
    effective_config = SimpleNamespace(
        host=fake_config.host,
        agents={"RecoveredChild": object()},
    )
    events: list[str] = []

    class _Manager:
        init_failures = []

        def set_agent_registration_hook(self, _hook) -> None:
            return None

        def reconcile_spawn_authority_restart_roster(self, config):
            assert config is fake_config
            events.append("reconcile")
            return effective_config

        async def load_from_config(
            self,
            config,
            *,
            restart_roster_reconciled,
        ) -> int:
            assert config is effective_config
            # Scheduler preflight seeded authority from this exact snapshot.
            # Loading must not re-read multi_agent.toml behind that authority
            # boundary and silently switch to a different tenant roster.
            assert restart_roster_reconciled is True
            events.append("load")
            return 0

        def list_agents(self):
            return {}

    manager = _Manager()

    async def _preflight(app, supplied_manager, config) -> None:
        assert supplied_manager is manager
        assert config is effective_config
        events.append("preflight")

    async def _start(app, supplied_manager, config) -> None:
        assert supplied_manager is manager
        assert config is effective_config
        events.append("host-start")

    shared_backend = object()

    async def _shared_backend(_app):
        return shared_backend

    def _manager_factory(**kwargs):
        assert kwargs["shared_postgres_backend"] is shared_backend
        return manager

    monkeypatch.setenv("KESTREL_MULTI_AGENT", "1")
    monkeypatch.setenv("KESTREL_API_KEY", "scheduler-host-test-key")
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "0")
    monkeypatch.setattr(server, "resolve_multi_agent_path", lambda _env: config_path)
    monkeypatch.setattr(ma_config.MultiAgentConfig, "load", lambda *_a, **_k: fake_config)
    monkeypatch.setattr(agent_manager, "AgentManager", _manager_factory)
    monkeypatch.setattr(
        server, "_prepare_shared_postgres_scheduler_protocol", _preflight
    )
    monkeypatch.setattr(
        server, "_start_shared_agent_postgres_backend", _shared_backend
    )
    monkeypatch.setattr(server, "_start_host_scheduler", _start)
    monkeypatch.setattr(did_registry, "install_a2a_did_resolver", lambda *_a, **_k: None)
    monkeypatch.setattr(phoenix_module, "should_supervise_phoenix", lambda: False)
    monkeypatch.setattr(demo_isolation, "classify_server_mode", lambda _agents: False)
    monkeypatch.setattr(server, "_mount_feature_ui_assets", lambda _app: None)
    monkeypatch.setattr(server, "_mount_feature_routers", lambda _app: None)
    monkeypatch.setattr(server, "setup_tracing", lambda _app: None)
    monkeypatch.setattr(hf, "instantiate_host_features", lambda **_k: [])

    async with server._lifespan_startup(FastAPI()):
        pass

    assert events == ["reconcile", "preflight", "load", "host-start"]
