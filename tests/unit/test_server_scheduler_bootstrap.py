"""Host scheduler protocol bootstrap sequencing regressions."""

from __future__ import annotations

import asyncio
import os
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
    """Hold state exists before preflight, agent load, and the host runner."""
    from kestrel_sovereign import host_features as hf
    from kestrel_sovereign.a2a import did_registry
    from kestrel_sovereign.multi_agent import agent_manager, config as ma_config
    from kestrel_sovereign import phoenix_supervisor as phoenix_module
    from kestrel_sovereign.security import demo_isolation

    runtime_base = tmp_path / "runtime-project"
    runtime_base.mkdir()
    config_dir = tmp_path / "external-config"
    config_dir.mkdir()
    config_path = config_dir / "multi_agent.toml"
    root_config = ma_config.LocalAgentConfig(
        data_dir=tmp_path / "root",
        port=8898,
    )
    cold_root_config = ma_config.LocalAgentConfig(
        data_dir=tmp_path / "cold-root",
        port=8897,
        autostart=False,
    )
    fake_config = ma_config.MultiAgentConfig(
        agents={"Root": root_config, "ColdRoot": cold_root_config},
    )
    fake_config.host.port = 8888
    fake_config.save(config_path)
    effective_config = fake_config.model_copy(deep=True)
    effective_config.agents["RecoveredChild"] = ma_config.LocalAgentConfig(
        data_dir=tmp_path / "recovered-child",
        port=8896,
    )
    events: list[str] = []
    removal_resolution_started = asyncio.Event()
    allow_removal_resolution = asyncio.Event()
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

        def __init__(self) -> None:
            self.created_agent_persistence_hook = None
            self.created_agent_registration_removal_hook = None

        def set_agent_registration_hook(self, _hook) -> None:
            return None

        def set_created_agent_persistence_hook(self, hook) -> None:
            self.created_agent_persistence_hook = hook

        def set_created_agent_registration_removal_hook(self, hook) -> None:
            self.created_agent_registration_removal_hook = hook

        def reconcile_spawn_authority_restart_roster(self, config):
            assert config is fake_config
            events.append("reconcile")
            return effective_config

        def bind_shared_postgres_backend(self, backend) -> None:
            assert backend is shared_backend
            events.append("backend-bind")

        def bind_hold_store(self, store) -> None:
            assert store is hold_store
            events.append("hold-bind")

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

        async def resolve_registered_agent_id(
            self,
            name,
            agent_config,
            *,
            require_config_identity=False,
        ) -> str:
            if name == "Root":
                assert require_config_identity is True
                assert agent_config == root_config
                return "did:test:root"
            if name == "ColdRoot":
                assert require_config_identity is True
                assert agent_config == cold_root_config
                return "did:test:cold-root"
            assert require_config_identity is True
            assert name == "PersistentChild"
            assert agent_config.port == 8899
            removal_resolution_started.set()
            await allow_removal_resolution.wait()
            return "did:test:persistent-child"

        def list_agents(self):
            return {}

        async def shutdown_all(self) -> None:
            return None

    manager = _Manager()

    async def _preflight(app, supplied_manager, config) -> None:
        assert supplied_manager is manager
        assert config is effective_config
        events.append("preflight")

    async def _stop_receipts(app) -> None:
        events.append("stop-receipts")
        app.state.stop_receipt_store = object()
        app.state.stop_receipt_db = None

    async def _start(app, supplied_manager, config) -> None:
        assert supplied_manager is manager
        assert config is effective_config
        events.append("host-start")

    async def _build_host_context(*, config):
        assert isinstance(config, dict)
        assert config["agents"] == ["Root", "ColdRoot", "RecoveredChild"]
        events.append("context-build")
        return host_context

    shared_backend = object()

    async def _shared_backend(_app):
        events.append("backend-start")
        return shared_backend

    def _manager_factory(**kwargs):
        assert "shared_postgres_backend" not in kwargs
        assert kwargs["base_data_dir"] == runtime_base
        return manager

    def _load_config(*_args, **kwargs):
        assert kwargs["runtime_base"] == runtime_base
        assert kwargs["runtime_env"] is os.environ
        return fake_config

    monkeypatch.chdir(runtime_base)
    monkeypatch.setenv("KESTREL_MULTI_AGENT", "1")
    monkeypatch.setenv("KESTREL_API_KEY", "scheduler-host-test-key")
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "0")
    monkeypatch.setattr(server, "resolve_multi_agent_path", lambda _env: config_path)
    monkeypatch.setattr(ma_config.MultiAgentConfig, "load", _load_config)
    monkeypatch.setattr(agent_manager, "AgentManager", _manager_factory)
    monkeypatch.setattr(
        server, "_prepare_shared_postgres_scheduler_protocol", _preflight
    )
    monkeypatch.setattr(
        server, "_start_shared_agent_postgres_backend", _shared_backend
    )
    monkeypatch.setattr(server, "_initialize_stop_receipts", _stop_receipts)
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
        child_config = ma_config.LocalAgentConfig(
            data_dir=tmp_path / "persistent-child",
            port=8899,
        )
        with pytest.raises(RuntimeError, match="restart-registered authority"):
            await manager.created_agent_persistence_hook(
                "Orphan",
                child_config,
                (("MissingParent", "did:test:missing-parent"),),
            )
        assert "Orphan" not in ma_config.MultiAgentConfig.from_file(
            config_path
        ).agents
        with pytest.raises(RuntimeError, match="autostart"):
            await manager.created_agent_persistence_hook(
                "ColdOrphan",
                child_config,
                (("ColdRoot", "did:test:cold-root"),),
            )
        assert "ColdOrphan" not in ma_config.MultiAgentConfig.from_file(
            config_path
        ).agents
        await manager.created_agent_persistence_hook(
            "PersistentChild",
            child_config,
            (("Root", "did:test:root"),),
        )
        # The manager writes the exact child row before publication so a crash
        # cannot leave a discoverable identity without desired startup state.
        # The later feature-level durability commit must treat that row as the
        # same transaction, not reject its own prepublication write.
        await manager.created_agent_persistence_hook(
            "PersistentChild",
            child_config,
            (("Root", "did:test:root"),),
        )
        removal_task = asyncio.create_task(
            manager.created_agent_registration_removal_hook(
                "PersistentChild",
                "did:test:persistent-child",
            )
        )
        await asyncio.wait_for(removal_resolution_started.wait(), timeout=1)
        external_config = ma_config.LocalAgentConfig(
            data_dir=tmp_path / "external-child",
            port=8901,
        )
        externally_edited = ma_config.MultiAgentConfig.from_file(config_path)
        externally_edited.agents["ExternalChild"] = external_config
        externally_edited.save(config_path)
        concurrent_config = ma_config.LocalAgentConfig(
            data_dir=tmp_path / "concurrent-child",
            port=8900,
        )
        concurrent_persist = asyncio.create_task(
            manager.created_agent_persistence_hook(
                "ConcurrentChild",
                concurrent_config,
                (("Root", "did:test:root"),),
            )
        )
        await asyncio.sleep(0)
        assert concurrent_persist.done() is False
        allow_removal_resolution.set()
        rollback = await removal_task
        await concurrent_persist
        assert "PersistentChild" not in ma_config.MultiAgentConfig.from_file(
            config_path
        ).agents
        assert ma_config.MultiAgentConfig.from_file(config_path).agents[
            "ConcurrentChild"
        ] == concurrent_config
        assert ma_config.MultiAgentConfig.from_file(config_path).agents[
            "ExternalChild"
        ] == external_config
        await rollback()
        restored = ma_config.MultiAgentConfig.from_file(config_path).agents
        assert restored["PersistentChild"] == child_config
        assert restored["ConcurrentChild"] == concurrent_config
        assert restored["ExternalChild"] == external_config

    assert events == [
        "stop-receipts",
        "reconcile",
        "context-build",
        "hold-bind",
        "backend-start",
        "backend-bind",
        "preflight",
        "load",
        "host-start",
    ]
    assert callable(manager.created_agent_persistence_hook)
    assert callable(manager.created_agent_registration_removal_hook)
    assert app.state.host_context is host_context
    assert app.state.host_context.hold_store is hold_store
    assert app.state.multi_agent_runtime_base == runtime_base
    assert app.state.multi_agent_runtime_env is os.environ


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

    async def _initialize_stop_receipts(_app) -> None:
        return None

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
    monkeypatch.setattr(server, "_initialize_stop_receipts", _initialize_stop_receipts)
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
